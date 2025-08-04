FROM runpod/base:0.6.3-cuda11.8.0

# Set python3.11 as the default python
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create model cache directories
RUN mkdir -p /runpod-volume/models/checkpoints && \
    mkdir -p /runpod-volume/models/loras && \
    mkdir -p /runpod-volume/models/jib-df

# Install dependencies
COPY requirements.txt /requirements.txt
RUN uv pip install -r /requirements.txt --system --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --index-strategy unsafe-best-match

# -------- CONVERT SDXL SAFETENSOR TO DIFFUSERS FORMAT --------
# Yeni (mevcut) script'i çek – isterseniz v0.34.0 tag'ine sabitleyin
ADD https://raw.githubusercontent.com/huggingface/diffusers/v0.34.0/scripts/convert_original_stable_diffusion_to_diffusers.py /tmp/convert_sdxl.py

# Download checkpoint if not exists and convert to Diffusers format
RUN python -c "\
import os; \
import requests; \
from pathlib import Path; \
CHECKPOINT_URL = 'https://civitai.com/api/download/models/1590699?type=Model&format=SafeTensor&size=pruned&fp=fp16'; \
CHECKPOINT_PATH = '/runpod-volume/models/checkpoints/jib_mix_illustrious_realistic_v2.safetensors'; \
checkpoint_file = Path(CHECKPOINT_PATH); \
print('Checking checkpoint...'); \
if not checkpoint_file.exists(): \
    print('Downloading checkpoint...'); \
    headers = {}; \
    civitai_key = os.environ.get('CIVITAI_API_KEY'); \
    if civitai_key: headers['Authorization'] = f'Bearer {civitai_key}'; \
    response = requests.get(CHECKPOINT_URL, headers=headers, stream=True); \
    response.raise_for_status(); \
    with open(CHECKPOINT_PATH, 'wb') as f: \
        for chunk in response.iter_content(chunk_size=8192): \
            if chunk: f.write(chunk); \
    print('Checkpoint downloaded successfully!'); \
else: \
    print('Checkpoint already exists, skipping download.'); \
"

# Convert checkpoint to Diffusers format
RUN python /tmp/convert_sdxl.py \
    --checkpoint_path /runpod-volume/models/checkpoints/jib_mix_illustrious_realistic_v2.safetensors \
    --dump_path /runpod-volume/models/jib-df \
    --pipeline_class_name StableDiffusionXLPipeline \
    --extract_ema && \
    echo "Checkpoint converted to Diffusers format successfully!" && \
    rm /tmp/convert_sdxl.py

# Add files
ADD handler.py .

# Set environment variables
ENV MODEL_CACHE_DIR=/runpod-volume/models
ENV CHECKPOINT_DIR=/runpod-volume/models/checkpoints
ENV LORA_DIR=/runpod-volume/models/loras
ENV HF_HOME=/runpod-volume/models/huggingface

# Run the handler
CMD python -u /handler.py
