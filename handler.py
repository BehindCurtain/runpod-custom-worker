"""Stable Diffusion XL Image Generation Handler with LoRA support."""

import os
import base64
import requests
import torch
import numpy as np
from io import BytesIO
from pathlib import Path
import runpod
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from peft import PeftModel

# Model configuration
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/runpod-volume/models")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/runpod-volume/models/checkpoints")
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

def setup_models():
    """Setup and download all required models."""
    print("Setting up models...")
    
    # Ensure checkpoint exists
    checkpoint_path = ensure_model_exists(CHECKPOINT_CONFIG, CHECKPOINT_DIR)
    
    # Ensure all LoRAs exist
    lora_paths = {}
    for lora_config in LORA_CONFIGS:
        lora_path = ensure_model_exists(lora_config, LORA_DIR)
        lora_paths[lora_config["name"]] = lora_path
    
    return checkpoint_path, lora_paths

def load_pipeline_with_loras(lora_mode="multi"):
    """Load and configure the Stable Diffusion XL pipeline with dynamic LoRA loading."""
    print(f"Loading Stable Diffusion XL pipeline with LoRA mode: {lora_mode}")
    
    checkpoint_path, lora_paths = setup_models()
    
    # Load the main pipeline
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(checkpoint_path),
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    
    # Move to GPU
    pipe = pipe.to("cuda")
    
    loaded_loras = []
    failed_loras = []
    
    # Determine which LoRAs to load based on mode
    if lora_mode == "none":
        print("LoRA mode: none - Skipping all LoRA loading")
        loras_to_load = []
    elif lora_mode == "single":
        print("LoRA mode: single - Loading only Detail Tweaker XL")
        loras_to_load = [LORA_CONFIGS[0]]  # Detail Tweaker XL
    elif lora_mode == "multi":
        print("LoRA mode: multi - Loading all LoRAs")
        loras_to_load = LORA_CONFIGS
    else:
        print(f"Unknown LoRA mode: {lora_mode}, defaulting to multi")
        loras_to_load = LORA_CONFIGS
    
    # Load LoRAs based on selected mode
    if loras_to_load:
        print(f"Loading {len(loras_to_load)} LoRA(s)...")
        
        for lora_config in loras_to_load:
            lora_path = lora_paths[lora_config["name"]]
            adapter_name = lora_config["name"].replace(" ", "_").replace("-", "_").lower()
            
            try:
                # Try different LoRA loading methods
                success = False
                
                # Method 1: Standard diffusers LoRA loading
                try:
                    pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
                    loaded_loras.append({
                        "name": lora_config["name"],
                        "adapter_name": adapter_name,
                        "scale": lora_config["scale"]
                    })
                    print(f"✓ Loaded LoRA: {lora_config['name']} with scale {lora_config['scale']}")
                    success = True
                except Exception as e1:
                    print(f"Method 1 failed for {lora_config['name']}: {e1}")
                    
                    # Method 2: Try loading as PEFT adapter
                    try:
                        from safetensors.torch import load_file
                        lora_weights = load_file(str(lora_path))
                        # This is a fallback - we'll skip PEFT for now and continue with base model
                        print(f"LoRA weights loaded but PEFT integration skipped for {lora_config['name']}")
                        success = False
                    except Exception as e2:
                        print(f"Method 2 failed for {lora_config['name']}: {e2}")
                
                if not success:
                    failed_loras.append(lora_config["name"])
                    
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
    
    try:
        # Use sequential CPU offload instead of model CPU offload for better stability
        pipe.enable_sequential_cpu_offload()
        print("✓ Sequential CPU offload enabled")
    except Exception as e:
        print(f"⚠ Could not enable CPU offload: {e}")
        try:
            pipe.enable_model_cpu_offload()
            print("✓ Model CPU offload enabled (fallback)")
        except Exception as e2:
            print(f"⚠ Could not enable any CPU offload: {e2}")
    
    print("Pipeline loaded successfully!")
    return pipe, loaded_loras, failed_loras

def load_pipeline():
    """Load and configure the Stable Diffusion XL pipeline with default multi LoRA mode."""
    return load_pipeline_with_loras("multi")

# Load pipeline globally
print("Initializing Stable Diffusion XL pipeline...")
pipeline, loaded_loras, failed_loras = load_pipeline()

def handler(job):
    """Handler function for image generation."""
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
        
        # LoRA mode parameter (new feature)
        lora_mode = job_input.get("lora_mode", "multi")
        print(f"LoRA mode requested: {lora_mode}")
        
        # Check if we need to reload pipeline with different LoRA mode
        global pipeline, loaded_loras, failed_loras
        current_lora_mode = getattr(pipeline, '_lora_mode', 'multi')
        
        if lora_mode != current_lora_mode:
            print(f"Reloading pipeline: {current_lora_mode} → {lora_mode}")
            pipeline, loaded_loras, failed_loras = load_pipeline_with_loras(lora_mode)
            pipeline._lora_mode = lora_mode  # Store current mode
        else:
            print(f"Using existing pipeline with LoRA mode: {lora_mode}")
        
        print(f"Generating image with prompt: {prompt[:100]}...")
        
        # Set up generator for reproducible results
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Generate image with improved error handling
        try:
            with torch.autocast("cuda"):
                result = pipeline(
                    prompt=prompt,
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
                    "lora_mode": lora_mode,
                    "loras_loaded": actual_loras,
                    "loras_failed": failed_loras if failed_loras else [],
                    "total_loras_attempted": len(LORA_CONFIGS) if lora_mode != "none" else 0,
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
