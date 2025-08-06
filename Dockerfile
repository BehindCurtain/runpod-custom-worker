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

# -------- TEMPLATE SYSTEM BUILD --------
# Copy template system files
COPY templates.json .
COPY templates.py .
COPY template_manager.py .
COPY build_all_templates.py /tmp/build_all_templates.py

# Patch'li conversion script'i kullan (fp16 variant desteği ile)
COPY convert_original_stable_diffusion_to_diffusers.py /tmp/convert_sdxl.py

# Build all templates (download and convert all unique checkpoints)
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
RUN python /tmp/build_all_templates.py \
    && echo "=== TEMPLATE BUILD COMPLETED ===" \
    && echo "=== LISTING ALL CONVERTED MODELS ===" \
    && find /app/models -type d -name "*-df" -exec echo "Found Diffusers model: {}" \; \
    && find /app/models -type f -name "model_index.json" -exec echo "Model index: {}" \; \
    && echo "=== CHECKING FOR FP16 VARIANT FILES ===" \
    && find /app/models -name "*fp16*" -exec ls -lh {} \; || echo "NO FP16 VARIANT FILES FOUND!" \
    && echo "=== TEMPLATE BUILD DEBUG COMPLETE ===" \
    && rm /tmp/build_all_templates.py /tmp/convert_sdxl.py

# Add handler file
ADD handler.py .

# Set environment variables
ENV MODEL_CACHE_DIR=/runpod-volume/models
ENV CHECKPOINT_DIR=/runpod-volume/models/checkpoints
ENV LORA_DIR=/runpod-volume/models/loras
ENV HF_HOME=/runpod-volume/models/huggingface

# Run the handler
CMD python -u /handler.py
