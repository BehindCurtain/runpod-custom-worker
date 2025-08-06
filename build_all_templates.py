#!/usr/bin/env python3
"""
Build-time script to download and convert all unique checkpoints from templates.
This ensures all templates have their required models available at runtime.
"""

import os
import sys
import subprocess
from pathlib import Path
from templates import get_all_unique_checkpoints, get_all_unique_loras
from template_manager import ensure_model_exists, ensure_all_template_loras

# Build-time directories
CHECKPOINT_DIR = "/runpod-volume/models/checkpoints"
DIFFUSERS_BUILD_DIR = "/app/models"

def download_checkpoint(checkpoint_config):
    """Download checkpoint if it doesn't exist."""
    print(f"=== DOWNLOADING CHECKPOINT: {checkpoint_config['name']} ===")
    
    checkpoint_path = Path(CHECKPOINT_DIR) / checkpoint_config["filename"]
    
    if not checkpoint_path.exists():
        print(f"Checkpoint {checkpoint_config['name']} not found, downloading...")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # Use requests to download with Civitai API key support
        import requests
        
        headers = {}
        civitai_api_key = os.environ.get('CIVITAI_API_KEY')
        if civitai_api_key:
            headers['Authorization'] = f'Bearer {civitai_api_key}'
            print(f'Using Civitai API key: {civitai_api_key[:8]}...{civitai_api_key[-4:]}')
        
        print(f"Downloading from: {checkpoint_config['url']}")
        response = requests.get(checkpoint_config['url'], headers=headers, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(checkpoint_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"Progress: {progress:.1f}%")
        
        print(f'✓ Checkpoint {checkpoint_config["name"]} downloaded successfully!')
    else:
        print(f'✓ Checkpoint {checkpoint_config["name"]} already exists, skipping download.')
    
    return checkpoint_path

def convert_checkpoint_to_diffusers(checkpoint_config, checkpoint_path):
    """Convert checkpoint to Diffusers format using patched conversion script."""
    diffusers_output_dir = Path(DIFFUSERS_BUILD_DIR) / checkpoint_config["diffusers_dir"]
    
    # Check if already converted
    model_index_path = diffusers_output_dir / "model_index.json"
    if model_index_path.exists():
        print(f"✓ Diffusers format already exists for {checkpoint_config['name']}, skipping conversion.")
        return diffusers_output_dir
    
    print(f"=== CONVERTING CHECKPOINT TO DIFFUSERS: {checkpoint_config['name']} ===")
    print(f"Input: {checkpoint_path}")
    print(f"Output: {diffusers_output_dir}")
    
    # Ensure output directory exists
    os.makedirs(diffusers_output_dir.parent, exist_ok=True)
    
    # Run conversion with patched script
    conversion_cmd = [
        "python", "/tmp/convert_sdxl.py",
        "--checkpoint_path", str(checkpoint_path),
        "--dump_path", str(diffusers_output_dir),
        "--pipeline_class_name", "StableDiffusionXLPipeline",
        "--extract_ema",
        "--from_safetensors",
        "--to_safetensors",
        "--half"  # Enable fp16 variant creation
    ]
    
    print(f"Running conversion command: {' '.join(conversion_cmd)}")
    
    try:
        # Set environment variable for conversion
        env = os.environ.copy()
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        
        result = subprocess.run(conversion_cmd, check=True, capture_output=True, text=True, env=env)
        print("=== CONVERSION OUTPUT ===")
        print(result.stdout)
        if result.stderr:
            print("=== CONVERSION STDERR ===")
            print(result.stderr)
        
        # Verify conversion success
        if model_index_path.exists():
            print(f"✓ Conversion successful for {checkpoint_config['name']}")
            
            # Debug: List generated files
            print("=== GENERATED FILES ===")
            for file in diffusers_output_dir.rglob("*"):
                if file.is_file():
                    print(f"  {file.relative_to(diffusers_output_dir)} ({file.stat().st_size} bytes)")
            
            # Check for fp16 variant files
            print("=== FP16 VARIANT FILES ===")
            fp16_files = list(diffusers_output_dir.rglob("*fp16*"))
            if fp16_files:
                for fp16_file in fp16_files:
                    print(f"  ✓ {fp16_file.relative_to(diffusers_output_dir)}")
            else:
                print("  ⚠ No fp16 variant files found!")
            
            return diffusers_output_dir
        else:
            raise RuntimeError(f"Conversion failed - model_index.json not found at {model_index_path}")
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Conversion failed for {checkpoint_config['name']}: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise RuntimeError(f"Checkpoint conversion failed: {e}")

def main():
    """Main build process for all templates."""
    print("=== BUILDING ALL TEMPLATES ===")
    
    # Get all unique checkpoints and LoRAs
    unique_checkpoints = get_all_unique_checkpoints()
    unique_loras = get_all_unique_loras()
    
    print(f"Found {len(unique_checkpoints)} unique checkpoints")
    print(f"Found {len(unique_loras)} unique LoRAs")
    
    # Download and convert all unique checkpoints
    for filename, checkpoint_config in unique_checkpoints.items():
        print(f"\n=== PROCESSING CHECKPOINT: {checkpoint_config['name']} ===")
        
        # Download checkpoint
        checkpoint_path = download_checkpoint(checkpoint_config)
        
        # Convert to Diffusers format
        diffusers_path = convert_checkpoint_to_diffusers(checkpoint_config, checkpoint_path)
        
        print(f"✓ Checkpoint {checkpoint_config['name']} ready at {diffusers_path}")
    
    # Ensure all LoRAs are available (will be downloaded at runtime if needed)
    print(f"\n=== PREPARING LORAS ===")
    ensure_all_template_loras()
    
    print("\n=== BUILD COMPLETE ===")
    print(f"✓ {len(unique_checkpoints)} checkpoints converted to Diffusers format")
    print(f"✓ {len(unique_loras)} LoRAs prepared for runtime download")
    print("All templates are ready for use!")

if __name__ == "__main__":
    main()
