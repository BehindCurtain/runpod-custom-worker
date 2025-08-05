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

# Create model cache directories (volume will be mounted at runtime)
RUN mkdir -p /runpod-volume/models/checkpoints && \
    mkdir -p /runpod-volume/models/loras && \
    mkdir -p /app/models/jib-df

# Install dependencies
COPY requirements.txt /requirements.txt
RUN uv pip install -r /requirements.txt --system --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --index-strategy unsafe-best-match

# -------- CONVERT SDXL SAFETENSOR TO DIFFUSERS FORMAT --------
# Patch'li conversion script'i kullan (fp16 variant desteği ile)
COPY convert_original_stable_diffusion_to_diffusers.py /tmp/convert_sdxl.py

# Copy and run checkpoint download script
COPY download_checkpoint.py /tmp/download_checkpoint.py
RUN python /tmp/download_checkpoint.py

# Convert checkpoint to Diffusers format with fp16 variant support
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
RUN python /tmp/convert_sdxl.py \
        --checkpoint_path /runpod-volume/models/checkpoints/jib_mix_illustrious_realistic_v2.safetensors \
        --dump_path /app/models/jib-df \
        --pipeline_class_name StableDiffusionXLPipeline \
        --extract_ema \
        --from_safetensors \
        --to_safetensors \
        --half \
    && echo "Checkpoint converted with fp16 variant to /app/models/jib-df!" \
    && echo "Verifying fp16 variant files..." \
    && ls -la /app/models/jib-df/ | grep "\.fp16\.safetensors" || echo "No fp16 files found (unexpected)" \
    && echo "All files in converted directory:" \
    && ls -la /app/models/jib-df/ \
    && rm /tmp/convert_sdxl.py /tmp/download_checkpoint.py

# Add files
ADD handler.py .

# Set environment variables
ENV MODEL_CACHE_DIR=/runpod-volume/models
ENV CHECKPOINT_DIR=/runpod-volume/models/checkpoints
ENV LORA_DIR=/runpod-volume/models/loras
ENV HF_HOME=/runpod-volume/models/huggingface

# Run the handler
CMD python -u /handler.py
