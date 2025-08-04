"""Stable Diffusion XL Image Generation Handler with LoRA support."""

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

def sanitize_name(name):
    """LoRA adını hem dosya adı hem adapter adı için güvenli hale getirir."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", name.lower())

def tokenize_chunks(tokenizer, prompt, max_tokens=75):
    """
    Token-bazlı kesin bölme - 75 token limit (CLS için 1 token ayrılıyor)
    
    Args:
        tokenizer: CLIP tokenizer
        prompt: Input prompt text
        max_tokens: Maximum tokens per chunk (default 75, leaving 1 for CLS)
    
    Returns:
        list: List of chunk texts, each guaranteed to be ≤76 tokens
    """
    # Tokenize without truncation to get full token sequence
    tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
    token_ids = tokens.input_ids[0]  # Remove batch dimension
    
    print(f"Total tokens: {len(token_ids)}")
    
    # Split into chunks of max_tokens size
    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        print(f"Chunk {len(chunks)}: {len(chunk_ids)} tokens - {chunk_text[:50]}...")
    
    return chunks

def long_prompt_to_embedding(pipe, prompt: str, max_tokens: int = 75):
    """
    Encode arbitrary-length prompt by chunking into exact token-based chunks
    and blending hidden states to fit into 77-token CLIP limit.
    
    Args:
        pipe: StableDiffusionXL pipeline
        prompt: Input prompt text
        max_tokens: Maximum tokens per chunk (default 75, leaving 1 for CLS)
    
    Returns:
        tuple: (prompt_embeds, pooled_embeds) or (None, None) if standard encoding
    """
    import time
    start_time = time.time()
    
    # Tokenize to check if we need chunking
    tokens = pipe.tokenizer(prompt, return_tensors="pt", truncation=False)
    token_count = tokens.input_ids.shape[1]
    
    print(f"Prompt token count: {token_count}")
    
    # If prompt fits in 77 tokens, use standard encoding
    if token_count <= 77:
        print("Using standard encoding (≤77 tokens)")
        return None, None  # Let pipeline handle normally
    
    print(f"Using token-based chunk blend encoding for {token_count} tokens")
    
    # Create token-based chunks with exact control
    chunks = tokenize_chunks(pipe.tokenizer, prompt, max_tokens)
    
    print(f"Split into {len(chunks)} token-based chunks")
    
    # Encode each chunk using encode_prompt
    text_chunks = []
    pooled_chunks = []
    
    for i, chunk in enumerate(chunks):
        print(f"Encoding chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
        
        try:
            text_e, _, pooled_e, _ = pipe.encode_prompt(
                prompt=chunk,
                device=pipe.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )
            text_chunks.append(text_e)
            pooled_chunks.append(pooled_e)
            print(f"✓ Successfully encoded chunk {i+1}")
            
        except Exception as e:
            print(f"✗ Failed to encode chunk {i+1}: {e}")
            continue
    
    if not text_chunks:
        print("✗ All chunks failed to encode, falling back to original prompt")
        return None, None
    
    # Build 77-token tensor from chunks
    final_text = build_77_token_tensor(text_chunks)
    final_pooled = torch.mean(torch.stack(pooled_chunks), dim=0)
    
    encode_time = time.time() - start_time
    print(f"✓ Token-based chunk blend encoding completed in {encode_time:.2f}s")
    print(f"Final embedding shapes: text={final_text.shape}, pooled={final_pooled.shape}")
    
    return final_text, final_pooled

def build_77_token_tensor(text_chunks):
    """Build a 77-token tensor from text chunks."""
    # Keep CLS token from first chunk
    cls_token = text_chunks[0][:, :1, :]  # (1, 1, dim)
    
    # Collect and blend the rest of the tokens
    rest_tokens = []
    for chunk in text_chunks:
        rest_tokens.append(chunk[:, 1:, :])  # Skip CLS token
    
    # Concatenate all non-CLS tokens
    concatenated_rest = torch.cat(rest_tokens, dim=1)  # (1, total_tokens, dim)
    
    # Truncate to fit 76 positions (77 - 1 for CLS)
    if concatenated_rest.shape[1] > 76:
        truncated_rest = concatenated_rest[:, :76, :]
        print(f"Truncated from {concatenated_rest.shape[1]} to 76 tokens")
    else:
        truncated_rest = concatenated_rest
        # Pad if necessary
        if truncated_rest.shape[1] < 76:
            padding_size = 76 - truncated_rest.shape[1]
            padding = torch.zeros(1, padding_size, truncated_rest.shape[2], 
                                device=truncated_rest.device, dtype=truncated_rest.dtype)
            truncated_rest = torch.cat([truncated_rest, padding], dim=1)
    
    # Combine CLS + rest tokens
    final_embedding = torch.cat([cls_token, truncated_rest], dim=1)  # (1, 77, dim)
    
    return final_embedding

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
        # Use sanitized name as key for consistent adapter naming
        sanitized_key = sanitize_name(lora_config["name"])
        lora_paths[sanitized_key] = lora_path
    
    return checkpoint_path, lora_paths

def load_pipeline():
    """Load and configure the Stable Diffusion XL pipeline with all LoRAs."""
    print("Loading Stable Diffusion XL pipeline with all LoRAs...")
    
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
        
        print(f"Generating image with prompt: {prompt[:100]}...")
        
        # Check for long prompt and handle with chunk blending if needed
        use_long_prompt = job_input.get("use_long_prompt", True)  # Default enabled
        p_emb, p_pool = (None, None)
        n_emb, n_pool = (None, None)
        
        if use_long_prompt:
            # Handle main prompt
            p_emb, p_pool = long_prompt_to_embedding(pipeline, prompt)
            
            # Handle negative prompt if provided
            if negative_prompt:
                n_emb, n_pool = long_prompt_to_embedding(pipeline, negative_prompt)
        
        # Set up generator for reproducible results
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Prepare fallback parameters for pipeline
        prompt_arg = prompt if p_emb is None else None
        negative_prompt_arg = negative_prompt if negative_prompt and n_emb is None else None
        
        # Generate image with improved error handling
        try:
            with torch.autocast("cuda"):
                result = pipeline(
                    prompt=prompt_arg,
                    negative_prompt=negative_prompt_arg,
                    prompt_embeds=p_emb,
                    pooled_prompt_embeds=p_pool,
                    negative_prompt_embeds=n_emb,
                    negative_pooled_prompt_embeds=n_pool,
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
                    "vae": "madebyollin/sdxl-vae-fp16-fix",
                    "long_prompt_enabled": use_long_prompt,
                    "used_chunk_blend": p_emb is not None,
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
