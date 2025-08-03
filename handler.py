"""Stable Diffusion XL Image Generation Handler with LoRA support."""

import os
import base64
import requests
import torch
from io import BytesIO
from pathlib import Path
import runpod
from diffusers import StableDiffusionXLPipeline
from PIL import Image

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
    
    # Add Civitai API key if available
    headers = {}
    civitai_api_key = os.environ.get("CIVITAI_API_KEY")
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

def load_pipeline():
    """Load and configure the Stable Diffusion XL pipeline."""
    print("Loading Stable Diffusion XL pipeline...")
    
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
    
    # Load LoRAs
    print("Loading LoRAs...")
    adapter_names = []
    adapter_weights = []
    
    for lora_config in LORA_CONFIGS:
        lora_path = lora_paths[lora_config["name"]]
        adapter_name = lora_config["name"].replace(" ", "_").lower()
        
        try:
            pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
            adapter_names.append(adapter_name)
            adapter_weights.append(lora_config["scale"])
            print(f"Loaded LoRA: {lora_config['name']} with scale {lora_config['scale']}")
        except Exception as e:
            print(f"Failed to load LoRA {lora_config['name']}: {e}")
    
    # Set adapters
    if adapter_names:
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        print(f"Set {len(adapter_names)} LoRA adapters")
    
    # Enable memory efficient attention
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    
    print("Pipeline loaded successfully!")
    return pipe

# Load pipeline globally
print("Initializing Stable Diffusion XL pipeline...")
pipeline = load_pipeline()

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
        
        print(f"Generating image with prompt: {prompt[:100]}...")
        
        # Set up generator for reproducible results
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Generate image
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
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        print("Image generated successfully!")
        
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
                "loras": [{"name": lora["name"], "scale": lora["scale"]} for lora in LORA_CONFIGS]
            }
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
