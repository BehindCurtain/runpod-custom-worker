"""Stable Diffusion XL Image Generation Handler with Template System and True LPW-SDXL (Diffusers format only)."""

import os
import base64
import torch
import numpy as np
from io import BytesIO
import runpod
from PIL import Image
from template_manager import load_template_pipeline
from templates import DEFAULT_TEMPLATE, list_templates

# Global pipeline variables - will be loaded on first request
pipeline = None
current_template = None
loaded_loras = []
failed_loras = []

def get_or_load_pipeline(template_name=None):
    """Get existing pipeline or load new one for specified template."""
    global pipeline, current_template, loaded_loras, failed_loras
    
    # Use default template if none specified
    if template_name is None:
        template_name = DEFAULT_TEMPLATE
    
    # Check if we need to load/reload pipeline
    if pipeline is None or current_template != template_name:
        print(f"Loading pipeline for template: {template_name}")
        pipeline, template, loaded_loras, failed_loras = load_template_pipeline(template_name)
        current_template = template_name
        print(f"✓ Pipeline loaded for template: {template_name}")
    else:
        print(f"✓ Using cached pipeline for template: {template_name}")
    
    return pipeline, current_template, loaded_loras, failed_loras

def handler(job):
    """Handler function for template-based image generation with True LPW-SDXL unlimited prompt support."""
    try:
        job_input = job["input"]
        
        # Get template selection
        template_name = job_input.get("template", DEFAULT_TEMPLATE)
        
        # Special endpoint to list available templates
        if job_input.get("list_templates", False):
            return {
                "templates": list_templates(),
                "default_template": DEFAULT_TEMPLATE,
                "current_template": current_template
            }
        
        # Get prompt from input
        prompt = job_input.get("prompt", "")
        if not prompt:
            return {"error": "No prompt provided"}
        
        # Load pipeline for requested template
        try:
            pipe, template_name, template_loras, template_failed_loras = get_or_load_pipeline(template_name)
        except Exception as template_error:
            return {"error": f"Failed to load template '{template_name}': {str(template_error)}"}
        
        # Generation parameters
        negative_prompt = job_input.get("negative_prompt", "")
        steps = job_input.get("steps", 24)
        cfg_scale = job_input.get("cfg_scale", 4.5)
        seed = job_input.get("seed", 797935397)
        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)
        
        print(f"Generating image with template '{template_name}' and prompt: {prompt[:100]}...")
        
        # Check prompt length for logging
        try:
            token_count = len(pipe.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids[0])
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
                result = pipe(
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
            
            # Get template info for metadata
            from templates import get_template
            template_info = get_template(template_name)
            
            # Check if using merged model
            from merge_template_loras import check_merged_model_exists
            is_merged = check_merged_model_exists(template_name)
            
            # Prepare metadata with template and LoRA information
            actual_loras = []
            lora_status = "not_available"
            
            if template_loras:
                actual_loras = [{"name": lora["name"], "scale": lora["scale"], "status": lora.get("status", "loaded")} for lora in template_loras]
                if is_merged:
                    lora_status = "merged_into_checkpoint"
                else:
                    lora_status = "runtime_adapters"
            
            return {
                "image": img_base64,
                "metadata": {
                    "template": {
                        "name": template_name,
                        "display_name": template_info["name"],
                        "description": template_info["description"],
                        "model_type": "merged" if is_merged else "base"
                    },
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "sampler": "DPM++ 2M Karras",
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "clip_skip": 2,
                    "checkpoint": {
                        "name": template_info["checkpoint"]["name"],
                        "format": "Diffusers (merged with LoRAs)" if is_merged else "Diffusers (base model)",
                        "merged_loras": is_merged
                    },
                    "vae": "madebyollin/sdxl-vae-fp16-fix",
                    "pipeline": "True LPW-SDXL (Diffusers format)",
                    "long_prompt_support": "Unlimited tokens via True LPW-SDXL",
                    "loras": {
                        "status": lora_status,
                        "loras_configured": actual_loras,
                        "loras_failed": template_failed_loras if template_failed_loras else [],
                        "total_loras_configured": len(template_info["loras"]),
                        "total_loras_active": len(template_loras) if template_loras else 0,
                        "merged_at_build_time": is_merged
                    }
                }
            }
            
        except Exception as generation_error:
            print(f"Error during image generation: {generation_error}")
            raise generation_error
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
