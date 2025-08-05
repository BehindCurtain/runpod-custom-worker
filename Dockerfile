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

# Copy and run checkpoint download script
COPY download_checkpoint.py /tmp/download_checkpoint.py
RUN python /tmp/download_checkpoint.py

# Convert checkpoint to Diffusers format
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
RUN python /tmp/convert_sdxl.py \
    --checkpoint_path /runpod-volume/models/checkpoints/jib_mix_illustrious_realistic_v2.safetensors \
    --dump_path /runpod-volume/models/jib-df \
    --pipeline_class_name StableDiffusionXLPipeline \
    --extract_ema \
    --from_safetensors && \
    echo "Checkpoint converted!" && \
    rm /tmp/convert_sdxl.py /tmp/download_checkpoint.py

# Add files
ADD handler.py .

# Set environment variables
ENV MODEL_CACHE_DIR=/runpod-volume/models
ENV CHECKPOINT_DIR=/runpod-volume/models/checkpoints
ENV LORA_DIR=/runpod-volume/models/loras
ENV HF_HOME=/runpod-volume/models/huggingface

# Run the handler
CMD python -u /handler.py
