"""Template management system for RunPod Custom Worker."""

import os
import torch
import re
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from templates import get_template, DEFAULT_TEMPLATE, get_all_unique_loras
import requests

# Model directories
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/runpod-volume/models")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/runpod-volume/models/checkpoints")
LORA_DIR = os.environ.get("LORA_DIR", "/runpod-volume/models/loras")
DIFFUSERS_BASE_DIR = os.environ.get("DIFFUSERS_BASE_DIR", "/runpod-volume/models/diffusers")
DIFFUSERS_BUILD_BASE_DIR = "/app/models"

def sanitize_name(name):
    """LoRA adını hem dosya adı hem adapter adı için güvenli hale getirir."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", name.lower())

def download_file(url, filepath):
    """Download a file from URL to filepath with progress tracking."""
    print(f"Downloading {filepath.name}...")
    
    # Debug: Environment variable kontrolü
    civitai_api_key = os.environ.get("CIVITAI_API_KEY")
    print(f"CIVITAI_API_KEY found: {bool(civitai_api_key)}")
    if civitai_api_key:
        print(f"CIVITAI_API_KEY length: {len(civitai_api_key)}")
    
    # Add Civitai API key if available
    headers = {}
    if civitai_api_key:
        headers["Authorization"] = f"Bearer {civitai_api_key}"
        print("Using Civitai API key for authenticated download")
    
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"Progress: {progress:.1f}%")
    
    print(f"Downloaded {filepath.name} successfully!")

def ensure_model_exists(config, model_dir):
    """Ensure model exists, download if not."""
    filepath = Path(model_dir) / config["filename"]
    
    if not filepath.exists():
        print(f"Model {config['name']} not found, downloading...")
        os.makedirs(model_dir, exist_ok=True)
        download_file(config["url"], filepath)
    else:
        print(f"Model {config['name']} already exists, skipping download.")
    
    return filepath

def check_diffusers_format_exists(diffusers_dir_name):
    """Check if Diffusers format exists with volume mount shadowing protection."""
    diffusers_path = Path(DIFFUSERS_BASE_DIR) / diffusers_dir_name
    model_index_path = diffusers_path / "model_index.json"
    
    print(f"=== CHECKING DIFFUSERS FORMAT: {diffusers_dir_name} ===")
    print(f"Checking path: {diffusers_path}")
    print(f"Path exists: {diffusers_path.exists()}")
    
    # Check if already exists in volume
    if diffusers_path.exists() and model_index_path.exists():
        print(f"✓ Diffusers format found at {diffusers_path}")
        return str(diffusers_path)
    
    # Check if exists in build location (not shadowed by volume)
    build_path = Path(DIFFUSERS_BUILD_BASE_DIR) / diffusers_dir_name
    build_model_index = build_path / "model_index.json"
    
    if build_path.exists() and build_model_index.exists():
        print(f"✓ Diffusers format found at build location {build_path}")
        print(f"Copying to volume location {diffusers_path} to fix volume mount shadowing...")
        
        try:
            import shutil
            # Ensure parent directory exists
            os.makedirs(diffusers_path.parent, exist_ok=True)
            # Copy entire directory tree
            shutil.copytree(str(build_path), str(diffusers_path))
            print(f"✓ Successfully copied Diffusers format from {build_path} to {diffusers_path}")
            return str(diffusers_path)
        except Exception as copy_error:
            print(f"✗ Failed to copy Diffusers format: {copy_error}")
            # Fallback: use build location directly
            print(f"Using build location {build_path} directly as fallback")
            return str(build_path)
    
    print(f"✗ Diffusers format not found for {diffusers_dir_name}")
    return None

def setup_template_models(template_name):
    """Setup models for a specific template."""
    print(f"Setting up models for template: {template_name}")
    
    template = get_template(template_name)
    
    # Verify checkpoint's Diffusers format exists
    checkpoint_config = template["checkpoint"]
    diffusers_path = check_diffusers_format_exists(checkpoint_config["diffusers_dir"])
    
    if not diffusers_path:
        error_msg = f"✗ CRITICAL: Diffusers format not found for checkpoint {checkpoint_config['name']}"
        print(error_msg)
        raise RuntimeError(f"Diffusers format missing for template {template_name}")
    
    print(f"✓ Checkpoint Diffusers format found at: {diffusers_path}")
    
    # Ensure all template LoRAs exist
    lora_paths = {}
    for lora_config in template["loras"]:
        lora_path = ensure_model_exists(lora_config, LORA_DIR)
        # Use sanitized name as key for consistent adapter naming
        sanitized_key = sanitize_name(lora_config["name"])
        lora_paths[sanitized_key] = {
            "path": lora_path,
            "scale": lora_config["scale"],
            "name": lora_config["name"]
        }
    
    return diffusers_path, lora_paths

def load_template_pipeline(template_name=None):
    """Load True LPW-SDXL pipeline for specified template."""
    if template_name is None:
        template_name = DEFAULT_TEMPLATE
        print(f"No template specified, using default: {template_name}")
    
    print(f"Loading True LPW-SDXL pipeline for template: {template_name}")
    
    template = get_template(template_name)
    diffusers_path, lora_paths = setup_template_models(template_name)
    
    # Load ONLY from Diffusers format with LPW-SDXL
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            diffusers_path,
            torch_dtype=torch.float16,
            custom_pipeline="lpw_stable_diffusion_xl",
            use_safetensors=False
        )
        print(f"✓ True LPW-SDXL pipeline loaded from {diffusers_path}")
        
    except Exception as e:
        error_msg = f"✗ CRITICAL: Diffusers format loading failed: {e}"
        print(error_msg)
        raise RuntimeError(f"Diffusers format loading failed for template {template_name}: {e}")
    
    # Move to GPU
    pipe = pipe.to("cuda")
    
    # Load FP16-safe SDXL VAE to fix black image issue
    print("Loading FP16-safe SDXL VAE...")
    try:
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        ).to("cuda")
        pipe.vae = vae
        print("✓ FP16-safe VAE loaded successfully")
        
        # Enable VAE slicing for additional VRAM savings
        try:
            pipe.enable_vae_slicing()
            print("✓ VAE slicing enabled")
        except Exception as vae_slice_error:
            print(f"⚠ Could not enable VAE slicing: {vae_slice_error}")
            
    except Exception as vae_error:
        print(f"⚠ Could not load FP16-safe VAE, using default: {vae_error}")
        print("This may cause black image issues with fp16 precision")
    
    loaded_loras = []
    failed_loras = []
    
    # Load template-specific LoRAs
    print(f"Loading {len(template['loras'])} LoRA(s) for template {template_name}...")
    
    for lora_config in template["loras"]:
        sanitized_key = sanitize_name(lora_config["name"])
        lora_info = lora_paths[sanitized_key]
        adapter_name = sanitized_key
        
        try:
            # Standard diffusers LoRA loading with sanitized adapter name
            pipe.load_lora_weights(str(lora_info["path"]), adapter_name=adapter_name)
            loaded_loras.append({
                "name": lora_info["name"],
                "adapter_name": adapter_name,
                "scale": lora_info["scale"]
            })
            print(f"✓ Loaded LoRA: {lora_info['name']} → {adapter_name} with scale {lora_info['scale']}")
                
        except Exception as e:
            print(f"✗ Failed to load LoRA {lora_info['name']}: {e}")
            failed_loras.append(lora_info["name"])
    
    # Set adapters only for successfully loaded LoRAs
    if loaded_loras:
        try:
            # Debug: Check what adapters are actually available
            if hasattr(pipe, 'get_list_adapters'):
                available_adapters = pipe.get_list_adapters()
                print(f"=== ADAPTER DEBUG ===")
                print(f"Available adapters: {available_adapters}")
                
                # Count actual adapters from unet component (most reliable)
                if available_adapters and 'unet' in available_adapters:
                    unet_adapters = available_adapters['unet']
                    actual_adapter_count = len(unet_adapters)
                    print(f"UNet adapters: {unet_adapters}")
                    print(f"Actual adapter count: {actual_adapter_count}")
                    
                    # Use UNet adapter names (they control the main model behavior)
                    if actual_adapter_count >= len(loaded_loras):
                        adapter_names = unet_adapters[:len(loaded_loras)]
                        adapter_weights = [lora["scale"] for lora in loaded_loras]
                        
                        print(f"Using UNet adapter names: {adapter_names}")
                        print(f"With weights: {adapter_weights}")
                        
                        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                        print(f"✓ Set {len(adapter_names)} LoRA adapters successfully")
                        
                        # Update loaded_loras with actual adapter names for metadata
                        for i, lora in enumerate(loaded_loras):
                            if i < len(adapter_names):
                                lora["actual_adapter_name"] = adapter_names[i]
                    else:
                        print(f"⚠ Mismatch: {actual_adapter_count} UNet adapters available, {len(loaded_loras)} LoRAs loaded")
                        print("Continuing with base model only...")
                else:
                    print("⚠ Cannot find UNet adapters in pipeline")
                    print("Continuing with base model only...")
            else:
                print("⚠ Cannot get adapter list from pipeline")
                print("Continuing with base model only...")
                
        except Exception as e:
            print(f"✗ Failed to set adapters: {e}")
            print("Continuing with base model only...")
    else:
        print("⚠ No LoRAs loaded successfully, using base model only")
    
    if failed_loras:
        print(f"⚠ Failed LoRAs: {', '.join(failed_loras)}")
    
    # Enable memory efficient settings
    try:
        pipe.enable_attention_slicing()
        print("✓ Attention slicing enabled")
    except Exception as e:
        print(f"⚠ Could not enable attention slicing: {e}")
    
    # Enable VAE tiling for additional memory savings (diffusers 0.34+ feature)
    try:
        if hasattr(pipe, 'enable_vae_tiling'):
            pipe.enable_vae_tiling()
            print("✓ VAE tiling enabled")
    except Exception as e:
        print(f"⚠ Could not enable VAE tiling: {e}")
    
    # Smart CPU offload based on GPU memory
    try:
        # Check available GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            print(f"GPU memory: {gpu_memory:.1f} GB")
            
            # Only enable CPU offload for GPUs with less than 20GB VRAM
            if gpu_memory < 20:
                pipe.enable_model_cpu_offload()
                print("✓ Model CPU offload enabled (GPU memory < 20GB)")
            else:
                print("✓ Keeping models on GPU (sufficient VRAM available)")
    except Exception as e:
        print(f"⚠ Could not configure CPU offload: {e}")
    
    print(f"True LPW-SDXL pipeline loaded successfully for template: {template_name}")
    print("✓ Unlimited prompt support active - no 77 token limit!")
    
    return pipe, template, loaded_loras, failed_loras

def ensure_all_template_loras():
    """Ensure all LoRAs from all templates are downloaded (for build-time preparation)."""
    print("Ensuring all template LoRAs are available...")
    
    unique_loras = get_all_unique_loras()
    
    for filename, lora_config in unique_loras.items():
        ensure_model_exists(lora_config, LORA_DIR)
    
    print(f"✓ All {len(unique_loras)} unique LoRAs are available")
