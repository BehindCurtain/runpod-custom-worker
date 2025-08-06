"""
Template LoRA merging system for RunPod Custom Worker.
Merges LoRAs into base checkpoints for optimized runtime performance.
"""

import os
import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline
from templates import get_template, get_templates
import shutil

# Directories
DIFFUSERS_BUILD_BASE_DIR = "/app/models"
LORA_DIR = os.environ.get("LORA_DIR", "/runpod-volume/models/loras")

def get_merged_model_path(template_name):
    """Get the path for merged model of a template."""
    return Path(DIFFUSERS_BUILD_BASE_DIR) / f"{template_name}-merged-df"

def get_base_model_path(template):
    """Get the path for base checkpoint in Diffusers format."""
    return Path(DIFFUSERS_BUILD_BASE_DIR) / template["checkpoint"]["diffusers_dir"]

def check_merged_model_exists(template_name):
    """Check if merged model already exists for template."""
    merged_path = get_merged_model_path(template_name)
    model_index_path = merged_path / "model_index.json"
    
    exists = merged_path.exists() and model_index_path.exists()
    
    if exists:
        print(f"✓ Merged model exists for template '{template_name}' at {merged_path}")
    else:
        print(f"✗ Merged model not found for template '{template_name}' at {merged_path}")
    
    return exists

def ensure_lora_exists(lora_config):
    """Ensure LoRA file exists, download if needed."""
    from template_manager import ensure_model_exists
    
    lora_path = ensure_model_exists(lora_config, LORA_DIR)
    return lora_path

def merge_template_loras(template_name):
    """Merge all LoRAs for a template into the base checkpoint."""
    print(f"=== MERGING LORAS FOR TEMPLATE: {template_name} ===")
    
    template = get_template(template_name)
    base_model_path = get_base_model_path(template)
    merged_model_path = get_merged_model_path(template_name)
    
    # Check if base model exists
    if not base_model_path.exists():
        raise RuntimeError(f"Base model not found at {base_model_path}. Run base checkpoint conversion first.")
    
    print(f"Base model: {base_model_path}")
    print(f"Target merged model: {merged_model_path}")
    print(f"LoRAs to merge: {len(template['loras'])}")
    
    # Load base pipeline
    print("Loading base pipeline...")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(base_model_path),
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        print("✓ Base pipeline loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load base pipeline: {e}")
        raise RuntimeError(f"Failed to load base pipeline from {base_model_path}: {e}")
    
    # Move to GPU for faster processing
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("✓ Pipeline moved to GPU")
    
    # Merge each LoRA sequentially
    merged_count = 0
    for i, lora_config in enumerate(template["loras"]):
        lora_name = lora_config["name"]
        lora_scale = lora_config["scale"]
        
        print(f"\n--- Merging LoRA {i+1}/{len(template['loras'])}: {lora_name} (scale: {lora_scale}) ---")
        
        try:
            # Ensure LoRA exists
            lora_path = ensure_lora_exists(lora_config)
            print(f"LoRA path: {lora_path}")
            
            # Load LoRA weights
            print(f"Loading LoRA weights...")
            pipe.load_lora_weights(str(lora_path))
            print(f"✓ LoRA weights loaded")
            
            # Fuse LoRA into the pipeline with specified scale
            print(f"Fusing LoRA with scale {lora_scale}...")
            pipe.fuse_lora(lora_scale=lora_scale)
            print(f"✓ LoRA fused successfully")
            
            # Unload LoRA weights to free memory
            pipe.unload_lora_weights()
            print(f"✓ LoRA weights unloaded")
            
            merged_count += 1
            
        except Exception as e:
            print(f"✗ Failed to merge LoRA {lora_name}: {e}")
            print(f"Continuing with remaining LoRAs...")
            continue
    
    print(f"\n=== MERGE SUMMARY ===")
    print(f"Successfully merged: {merged_count}/{len(template['loras'])} LoRAs")
    
    if merged_count == 0:
        print("⚠ No LoRAs were merged successfully. Saving base model as merged model.")
    
    # Save merged model
    print(f"\nSaving merged model to {merged_model_path}...")
    try:
        # Ensure output directory exists
        os.makedirs(merged_model_path.parent, exist_ok=True)
        
        # Save with fp16 variant for consistency
        pipe.save_pretrained(
            str(merged_model_path),
            safe_serialization=True,
            variant="fp16"
        )
        print(f"✓ Merged model saved successfully")
        
        # Verify save was successful
        model_index_path = merged_model_path / "model_index.json"
        if model_index_path.exists():
            print(f"✓ Merge verification successful - model_index.json found")
            
            # List generated files for debugging
            print("=== MERGED MODEL FILES ===")
            for file in merged_model_path.rglob("*"):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"  {file.relative_to(merged_model_path)} ({size_mb:.1f} MB)")
            
            return str(merged_model_path)
        else:
            raise RuntimeError("Merge verification failed - model_index.json not found")
            
    except Exception as e:
        print(f"✗ Failed to save merged model: {e}")
        raise RuntimeError(f"Failed to save merged model: {e}")
    
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ GPU memory cleared")

def merge_all_templates():
    """Merge LoRAs for all templates."""
    print("=== MERGING ALL TEMPLATES ===")
    
    templates = get_templates()
    total_templates = len(templates)
    merged_templates = []
    failed_templates = []
    
    for i, template_name in enumerate(templates.keys()):
        print(f"\n=== PROCESSING TEMPLATE {i+1}/{total_templates}: {template_name} ===")
        
        try:
            # Check if already merged
            if check_merged_model_exists(template_name):
                print(f"✓ Template '{template_name}' already merged, skipping")
                merged_templates.append(template_name)
                continue
            
            # Merge template
            merged_path = merge_template_loras(template_name)
            merged_templates.append(template_name)
            print(f"✓ Template '{template_name}' merged successfully at {merged_path}")
            
        except Exception as e:
            print(f"✗ Failed to merge template '{template_name}': {e}")
            failed_templates.append(template_name)
            continue
    
    print(f"\n=== MERGE ALL TEMPLATES SUMMARY ===")
    print(f"Total templates: {total_templates}")
    print(f"Successfully merged: {len(merged_templates)}")
    print(f"Failed: {len(failed_templates)}")
    
    if merged_templates:
        print(f"Merged templates: {', '.join(merged_templates)}")
    
    if failed_templates:
        print(f"Failed templates: {', '.join(failed_templates)}")
        raise RuntimeError(f"Failed to merge {len(failed_templates)} templates")
    
    print("✓ All templates merged successfully!")
    return merged_templates

def cleanup_base_models():
    """Clean up base models after merging to save disk space (optional)."""
    print("=== CLEANING UP BASE MODELS ===")
    
    templates = get_templates()
    base_models_to_keep = set()
    
    # Collect all unique base models that are used
    for template in templates.values():
        base_model_path = get_base_model_path(template)
        base_models_to_keep.add(str(base_model_path))
    
    print(f"Base models to keep: {len(base_models_to_keep)}")
    for base_model in base_models_to_keep:
        print(f"  {base_model}")
    
    # Note: We keep base models as they might be needed for future template changes
    print("✓ Base models preserved for future use")

if __name__ == "__main__":
    """CLI interface for merging templates."""
    import sys
    
    if len(sys.argv) > 1:
        template_name = sys.argv[1]
        print(f"Merging specific template: {template_name}")
        
        try:
            if check_merged_model_exists(template_name):
                print(f"Template '{template_name}' already merged")
            else:
                merged_path = merge_template_loras(template_name)
                print(f"✓ Template '{template_name}' merged successfully at {merged_path}")
        except Exception as e:
            print(f"✗ Failed to merge template '{template_name}': {e}")
            sys.exit(1)
    else:
        print("Merging all templates...")
        try:
            merge_all_templates()
        except Exception as e:
            print(f"✗ Failed to merge all templates: {e}")
            sys.exit(1)
