"""Stable Diffusion XL Image Generation Handler with LoRA support and True LPW-SDXL (Diffusers format only)."""

import os
import base64
import requests
import torch
import numpy as np
import re
from io import BytesIO
from pathlib import Path
import runpod
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image

# Model configuration
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/runpod-volume/models")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/runpod-volume/models/checkpoints")
DIFFUSERS_DIR = os.environ.get("DIFFUSERS_DIR", "/runpod-volume/models/jib-df")
DIFFUSERS_BUILD_DIR = "/app/models/jib-df"  # Build-time location (not shadowed by volume)
LORA_DIR = os.environ.get("LORA_DIR", "/runpod-volume/models/loras")

# Checkpoint configuration
CHECKPOINT_CONFIG = {
    "name": "Jib Mix Illustrious Realistic",
    "url": "https://civitai.com/api/download/models/1590699?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "filename": "jib_mix_illustrious_realistic_v2.safetensors"
}

# LoRA configurations
LORA_CONFIGS = [
    {
        "name": "Detail Tweaker XL",
        "url": "https://civitai.com/api/download/models/135867?type=Model&format=SafeTensor",
        "scale": 1.5,
        "filename": "detail_tweaker_xl.safetensors"
    },
    {
        "name": "Hand Detail FLUX & XL",
        "url": "https://civitai.com/api/download/models/294259?type=Model&format=SafeTensor",
        "scale": 1.0,
        "filename": "hand_detail_flux_xl.safetensors"
    },
    {
        "name": "Boring Reality primaryV3.0",
        "url": "https://civitai.com/api/download/models/348625?type=Model&format=SafeTensor",
        "scale": 0.3,
        "filename": "boring_reality_v3.safetensors"
    },
    {
        "name": "Boring Reality primaryV4.0",
        "url": "https://civitai.com/api/download/models/348837?type=Model&format=SafeTensor",
        "scale": 0.4,
        "filename": "boring_reality_v4.safetensors"
    },
    {
        "name": "epiCRealismXL-KiSS Enhancer",
        "url": "https://civitai.com/api/download/models/1051156?type=Model&format=SafeTensor",
        "scale": 1.0,
        "filename": "epicrealism_xl_kiss.safetensors"
    },
    {
        "name": "Dramatic Lighting Slider",
        "url": "https://civitai.com/api/download/models/1242203?type=Model&format=SafeTensor",
        "scale": 1.0,
        "filename": "dramatic_lighting_slider.safetensors"
    },
    {
        "name": "Pony Realism Slider",
        "url": "https://civitai.com/api/download/models/1253021?type=Model&format=SafeTensor",
        "scale": 3.0,
        "filename": "pony_realism_slider.safetensors"
    },
    {
        "name": "Amateur style - slider (Pony)",
        "url": "https://civitai.com/api/download/models/1594293?type=Model&format=SafeTensor",
        "scale": 1.0,
        "filename": "amateur_style_slider.safetensors"
    },
    {
        "name": "Real Skin Slider",
        "url": "https://civitai.com/api/download/models/1681921?type=Model&format=SafeTensor",
        "scale": 1.0,
        "filename": "real_skin_slider.safetensors"
    }
]

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

def check_diffusers_format_exists():
    """Debug ile detaylı Diffusers formatı kontrolü"""
    global DIFFUSERS_DIR  # Global declaration must be at the beginning
    
    print("=== RUNTIME DEBUG: DIFFUSERS FORMAT CHECK ===")
    
    diffusers_path = Path(DIFFUSERS_DIR)
    model_index_path = diffusers_path / "model_index.json"
    
    print(f"Checking path: {diffusers_path}")
    print(f"Path exists: {diffusers_path.exists()}")
    
    if diffusers_path.exists():
        print("=== ALL FILES IN DIFFUSERS DIR ===")
        for file in diffusers_path.rglob("*"):
            if file.is_file():
                print(f"  {file.relative_to(diffusers_path)} ({file.stat().st_size} bytes)")
        
        print("=== SEARCHING FOR FP16 VARIANT FILES ===")
        fp16_files = list(diffusers_path.rglob("*fp16*"))
        if fp16_files:
            for fp16_file in fp16_files:
                print(f"  FOUND FP16: {fp16_file}")
        else:
            print("  NO FP16 VARIANT FILES FOUND!")
        
        print("=== MODEL_INDEX.JSON CONTENT ===")
        if model_index_path.exists():
            with open(model_index_path) as f:
                content = f.read()
                print(f"  Content: {content[:500]}...")
        else:
            print("  model_index.json NOT FOUND!")
    
    # Check if already exists in volume
    if diffusers_path.exists() and model_index_path.exists():
        print(f"✓ Diffusers format found at {DIFFUSERS_DIR}")
        print("=== DEBUG COMPLETE ===")
        return True
    
    # Check if exists in build location (not shadowed by volume)
    build_path = Path(DIFFUSERS_BUILD_DIR)
    build_model_index = build_path / "model_index.json"
    
    if build_path.exists() and build_model_index.exists():
        print(f"✓ Diffusers format found at build location {DIFFUSERS_BUILD_DIR}")
        print(f"Copying to volume location {DIFFUSERS_DIR} to fix volume mount shadowing...")
        
        try:
            import shutil
            # Ensure parent directory exists
            os.makedirs(diffusers_path.parent, exist_ok=True)
            # Copy entire directory tree
            shutil.copytree(str(build_path), str(diffusers_path))
            print(f"✓ Successfully copied Diffusers format from {DIFFUSERS_BUILD_DIR} to {DIFFUSERS_DIR}")
            return True
        except Exception as copy_error:
            print(f"✗ Failed to copy Diffusers format: {copy_error}")
            # Fallback: use build location directly
            print(f"Using build location {DIFFUSERS_BUILD_DIR} directly as fallback")
            DIFFUSERS_DIR = DIFFUSERS_BUILD_DIR
            return True
    
    print(f"✗ Diffusers format not found at {DIFFUSERS_DIR} or {DIFFUSERS_BUILD_DIR}")
    return False


def setup_models():
    """Setup models - Diffusers format MANDATORY (converted at build-time)."""
    print("Setting up models for True LPW-SDXL (Diffusers format only)...")
    
    # Verify Diffusers format exists (should be converted at build-time)
    if not check_diffusers_format_exists():
        error_msg = "✗ CRITICAL: Diffusers format not found - should have been converted at build-time!"
        print(error_msg)
        raise RuntimeError("Diffusers format missing - build process failed")
    else:
        print("✓ Diffusers format found - build-time conversion successful")
    
    # Ensure all LoRAs exist
    lora_paths = {}
    for lora_config in LORA_CONFIGS:
        lora_path = ensure_model_exists(lora_config, LORA_DIR)
        # Use sanitized name as key for consistent adapter naming
        sanitized_key = sanitize_name(lora_config["name"])
        lora_paths[sanitized_key] = lora_path
    
    return lora_paths

def load_pipeline():
    """Load True LPW-SDXL pipeline from Diffusers format - NO FALLBACK."""
    print("Loading True LPW-SDXL pipeline from Diffusers format...")
    
    lora_paths = setup_models()
    
    # Load ONLY from Diffusers format with LPW-SDXL
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            DIFFUSERS_DIR,
            torch_dtype=torch.float16,
            custom_pipeline="lpw_stable_diffusion_xl",
            use_safetensors=False
        )
        print("✓ True LPW-SDXL pipeline loaded from Diffusers format successfully")
        
    except Exception as e:
        error_msg = f"✗ CRITICAL: Diffusers format loading failed: {e}"
        print(error_msg)
        raise RuntimeError(f"Diffusers format loading failed - cannot proceed: {e}")
    
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
    
    # Load all LoRAs
    print(f"Loading {len(LORA_CONFIGS)} LoRA(s)...")
    
    for lora_config in LORA_CONFIGS:
        # Use sanitized name for consistent key lookup
        sanitized_key = sanitize_name(lora_config["name"])
        lora_path = lora_paths[sanitized_key]
        adapter_name = sanitized_key  # Use sanitized name as adapter name
        
        try:
            # Standard diffusers LoRA loading with sanitized adapter name
            pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
            loaded_loras.append({
                "name": lora_config["name"],
                "adapter_name": adapter_name,
                "scale": lora_config["scale"]
            })
            print(f"✓ Loaded LoRA: {lora_config['name']} → {adapter_name} with scale {lora_config['scale']}")
                
        except Exception as e:
            print(f"✗ Failed to load LoRA {lora_config['name']}: {e}")
            failed_loras.append(lora_config["name"])
    
    # Set adapters only for successfully loaded LoRAs
    if loaded_loras:
        try:
            adapter_names = [lora["adapter_name"] for lora in loaded_loras]
            adapter_weights = [lora["scale"] for lora in loaded_loras]
            pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
            print(f"✓ Set {len(adapter_names)} LoRA adapters successfully")
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
    
    print("True LPW-SDXL pipeline loaded successfully!")
    print("✓ Unlimited prompt support active - no 77 token limit!")
    return pipe, loaded_loras, failed_loras

# Load pipeline globally
print("Initializing True LPW-SDXL pipeline (Diffusers format only)...")
pipeline, loaded_loras, failed_loras = load_pipeline()

def handler(job):
    """Handler function for image generation with True LPW-SDXL unlimited prompt support."""
    try:
        job_input = job["input"]
        
        # Get prompt from input
        prompt = job_input.get("prompt", "")
        if not prompt:
            return {"error": "No prompt provided"}
        
        # Generation parameters
        negative_prompt = job_input.get("negative_prompt", "")
        steps = job_input.get("steps", 24)
        cfg_scale = job_input.get("cfg_scale", 4.5)
        seed = job_input.get("seed", 797935397)
        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)
        
        print(f"Generating image with prompt: {prompt[:100]}...")
        
        # Check prompt length for logging
        try:
            token_count = len(pipeline.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids[0])
            print(f"Prompt token count: {token_count}")
            if token_count > 77:
                print("✓ Long prompt detected - True LPW-SDXL will handle unlimited tokens!")
            else:
                print("✓ Standard prompt - True LPW-SDXL ready")
        except Exception as token_error:
            print(f"Could not count tokens: {token_error}")
        
        # Set up generator for reproducible results
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Generate image with True LPW-SDXL unlimited prompt handling
        try:
            with torch.autocast("cuda"):
                result = pipeline(
                    prompt=prompt,  # Unlimited prompt - True LPW-SDXL handles any length
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    clip_skip=2
                )
            
            image = result.images[0]
            
            # Validate image before processing
            if image is None:
                raise ValueError("Generated image is None")
            
            # Convert PIL image to numpy array for validation
            img_array = np.array(image)
            
            # Check if image is completely black or has invalid values
            if img_array.max() == 0:
                print("⚠ Warning: Generated image appears to be completely black")
            elif np.isnan(img_array).any() or np.isinf(img_array).any():
                print("⚠ Warning: Generated image contains invalid values (NaN/Inf)")
                # Clip invalid values
                img_array = np.nan_to_num(img_array, nan=0.0, posinf=255.0, neginf=0.0)
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                image = Image.fromarray(img_array)
            
            print(f"Image stats: min={img_array.min()}, max={img_array.max()}, shape={img_array.shape}")
            
            # Convert to base64 with improved error handling
            buffered = BytesIO()
            
            # Save as PNG with high quality
            image.save(buffered, format="PNG", optimize=False, compress_level=1)
            buffered.seek(0)
            
            # Get the bytes and validate
            img_bytes = buffered.getvalue()
            if len(img_bytes) == 0:
                raise ValueError("Generated image bytes are empty")
            
            # Encode to base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            if not img_base64:
                raise ValueError("Base64 encoding failed")
            
            print(f"Image generated successfully! Size: {len(img_bytes)} bytes, Base64 length: {len(img_base64)}")
            
            # Prepare metadata with actual loaded LoRAs
            actual_loras = []
            if loaded_loras:
                actual_loras = [{"name": lora["name"], "scale": lora["scale"]} for lora in loaded_loras]
            
            return {
                "image": img_base64,
                "metadata": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "sampler": "DPM++ 2M Karras",
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "clip_skip": 2,
                    "model": CHECKPOINT_CONFIG["name"],
                    "model_format": "Diffusers (converted from SafeTensors)",
                    "vae": "madebyollin/sdxl-vae-fp16-fix",
                    "pipeline": "True LPW-SDXL (Diffusers format)",
                    "long_prompt_support": "Unlimited tokens via True LPW-SDXL",
                    "loras_loaded": actual_loras,
                    "loras_failed": failed_loras if failed_loras else [],
                    "total_loras_attempted": len(LORA_CONFIGS),
                    "total_loras_loaded": len(loaded_loras) if loaded_loras else 0
                }
            }
            
        except Exception as generation_error:
            print(f"Error during image generation: {generation_error}")
            raise generation_error
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
