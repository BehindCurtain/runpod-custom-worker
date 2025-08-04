# RunPod Custom Worker - Modül Haritaları

## İş İşleme Sistemi Modülleri

### handler.py Modülü

**Amaç**: LPW-SDXL Community Pipeline kullanarak Stable Diffusion XL tabanlı görüntü üretimi ve LoRA yönetimi sağlar.

**Bileşenler**:
- `handler(job)` fonksiyonu - Ana görüntü üretim entry point
- `load_pipeline()` - LPW-SDXL community pipeline kurulumu
- `setup_models()` - Model indirme ve kontrol sistemi
- `download_file()` - Civitai model indirme
- `ensure_model_exists()` - Model varlık kontrolü
- `sanitize_name()` - LoRA adı sanitizasyonu

**Sorumluluklar**:
- Prompt ve parametrelerin işlenmesi
- Otomatik uzun prompt desteği (LPW-SDXL community pipeline ile)
- Checkpoint ve LoRA modellerinin yönetimi
- LoRA adı sanitizasyonu ve adapter yönetimi
- Bellek optimizasyonu (sequential CPU offload kaldırıldı)
- FP16-safe VAE yönetimi (siyah görüntü fix'i)
- Stable Diffusion XL inference
- Base64 görüntü dönüşümü
- Error handling ve logging

**LPW-SDXL Community Pipeline Sistemi**:
```python
def load_pipeline():
    """LPW-SDXL community pipeline ile otomatik uzun prompt desteği"""
    try:
        pipe = DiffusionPipeline.from_single_file(
            str(checkpoint_path),
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            custom_pipeline="lpw_stable_diffusion_xl"  # Otomatik uzun prompt
        )
        print("✓ LPW-SDXL community pipeline loaded successfully")
    except Exception as lpw_error:
        # Fallback to standard SDXL
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_single_file(...)

# Handler'da sadeleştirilmiş kullanım
result = pipeline(
    prompt=prompt,  # Doğrudan prompt - LPW-SDXL otomatik chunking
    negative_prompt=negative_prompt,
    num_inference_steps=steps,
    guidance_scale=cfg_scale,
    # ... diğer parametreler
)
```

**Bellek Optimizasyon Stratejisi**:
```python
# Korunan optimizasyonlar
pipe.enable_attention_slicing()  # ✅ Korundu
pipe.enable_vae_slicing()        # ✅ Korundu

# Kaldırılan problemli optimizasyon
# pipe.enable_sequential_cpu_offload()  # ❌ Meta tensor hatası nedeniyle kaldırıldı

# Yeni optimizasyonlar
pipe.enable_vae_tiling()  # ✅ Diffusers 0.34+ özelliği

# Akıllı CPU offload
if gpu_memory < 20:  # GB
    pipe.enable_model_cpu_offload()  # Sadece düşük VRAM'da
```

**Bağımlılıklar**:
- `runpod` - Serverless platform
- `diffusers` - Stable Diffusion pipeline
- `torch` - PyTorch backend
- `transformers` - Model transformers
- `PIL` - Görüntü işleme
- `requests` - Model indirme
- `peft` - LoRA adapter desteği
- `numpy` - Görüntü validasyonu
- `safetensors` - Güvenli model yükleme

**Model Konfigürasyonu**:
```python
# Checkpoint configuration
CHECKPOINT_CONFIG = {
    "name": "Jib Mix Illustrious Realistic",
    "url": "https://civitai.com/api/download/models/1590699...",
    "filename": "jib_mix_illustrious_realistic_v2.safetensors"
}

# LoRA configurations
LORA_CONFIGS = [
    {
        "name": "Detail Tweaker XL",
        "url": "https://civitai.com/api/download/models/135867...",
        "scale": 1.5,
        "filename": "detail_tweaker_xl.safetensors"
    },
    # ... 8 additional LoRAs
]

# Civitai API authentication
def download_file(url, filepath):
    headers = {}
    civitai_api_key = os.environ.get("CIVITAI_API_KEY")
    if civitai_api_key:
        headers["Authorization"] = f"Bearer {civitai_api_key}"
    response = requests.get(url, headers=headers, stream=True)
```

## Bağımlılık Yönetim Sistemi Modülleri

### requirements.txt Modülü

**Amaç**: Stable Diffusion XL ve AI/ML paket bağımlılıklarını tanımlar.

**Yapı**:
```
# Core dependency
runpod~=1.7.13

# Stable Diffusion and AI/ML dependencies
diffusers==0.34.*
torch==2.6.*+cu118
transformers==4.54.*
accelerate==1.9.*
peft==0.17.*
safetensors>=0.4.0
pillow>=10.0.0
requests>=2.31.0

# Hugging Face fast download support
hf_transfer>=0.1.4
```

**Versioning Stratejisi**:
- `~=`: RunPod SDK için compatible release
- `==x.*`: AI/ML kütüphaneleri için specific major.minor version
- `+cu118`: PyTorch için CUDA 11.8 specific build
- CUDA 11.8.0 compatibility sağlanmış

**Kritik Paketler**:
- `diffusers`: Stable Diffusion XL pipeline (v0.34.x)
- `torch`: GPU acceleration ve model loading (v2.6.x+cu118)
- `transformers`: Text encoder ve tokenizer (v4.54.x)
- `accelerate`: Memory efficient model loading (v1.9.x)
- `peft`: LoRA adapter yükleme ve yönetimi (v0.17.x)
- `safetensors`: Güvenli model format desteği

## Container Yönetim Sistemi Modülleri

### Dockerfile Modülü

**Amaç**: Docker container tanımını ve build sürecini yönetir.

**Katman Yapısı**:

1. **Base Layer**:
   ```dockerfile
   FROM runpod/base:0.6.3-cuda11.8.0
   ```
   - CUDA 11.8.0 desteği
   - Python multiple versions
   - uv package manager
   - Jupyter notebook

2. **Python Configuration Layer**:
   ```dockerfile
   RUN ln -sf $(which python3.11) /usr/local/bin/python
   ```
   - Python 3.11 default setup
   - Symlink management

3. **Dependencies Layer**:
   ```dockerfile
   COPY requirements.txt /requirements.txt
   RUN uv pip install -r /requirements.txt --system --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cu118
   ```
   - Package installation with CUDA 11.8 support
   - PyTorch CUDA-specific wheel installation
   - Cache optimization

4. **Application Layer**:
   ```dockerfile
   ADD handler.py .
   CMD python -u /handler.py
   ```
   - Code deployment
   - Execution setup

**Optimizasyon Noktaları**:
- Layer caching için dependency installation önce
- `--no-cache-dir` ile disk space optimization
- `--system` flag ile global installation

**Genişletme Şablonları**:

```dockerfile
# System dependencies eklemek için
RUN apt-get update && apt-get install -y \
    package-name \
    && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV MODEL_PATH=/models
ENV CUDA_VISIBLE_DEVICES=0

# Additional files
COPY models/ /models/
COPY config/ /config/
```

## Test ve Geliştirme Sistemi Modülleri

### test_input.json Modülü

**Amaç**: Yerel test için sample input verisi sağlar.

**Yapı**:
```json
{
  "input": {
    "name": "John Doe"
  }
}
```

**Genişletme Örnekleri**:

```json
// Text processing için
{
  "input": {
    "text": "Sample text to process",
    "max_length": 100,
    "temperature": 0.7
  }
}

// Image processing için
{
  "input": {
    "image_url": "https://example.com/image.jpg",
    "resize_width": 512,
    "resize_height": 512
  }
}

// Model inference için
{
  "input": {
    "prompt": "A beautiful sunset over mountains",
    "steps": 20,
    "guidance_scale": 7.5
  }
}
```

### Local Testing Infrastructure

**Test Execution Pattern**:
```bash
# Direct execution
python handler.py

# With custom input
echo '{"input": {"custom": "data"}}' > custom_input.json
python handler.py custom_input.json
```

## Modüller Arası İletişim

### Veri Akış Şemaları

1. **Development Flow**:
   ```
   test_input.json → handler.py → console output
   ```

2. **Build Flow**:
   ```
   requirements.txt → Dockerfile → Docker Image
   ```

3. **Runtime Flow**:
   ```
   RunPod Job → handler.py → RunPod Response
   ```

### Interface Definitions

**Job Input Interface**:
```python
{
    "input": {
        # User-defined parameters
        "param1": "value1",
        "param2": "value2"
    },
    "id": "job_id",  # RunPod tarafından sağlanır
    # Diğer RunPod metadata
}
```

**Handler Output Interface**:
```python
# Simple string response
return "Hello, World!"

# Structured response
return {
    "result": "processed_data",
    "metadata": {
        "processing_time": 1.23,
        "model_version": "v1.0"
    }
}

# Error response
return {
    "error": "Error message",
    "code": "ERROR_CODE"
}
```

## Modül Genişletme Rehberi

### Yeni Modül Ekleme

1. **Python modülü eklemek için**:
   - `src/` klasörü oluştur
   - Modülleri `src/` altına yerleştir
   - Dockerfile'da `COPY src/ /src/` ekle
   - `handler.py`'da import et

2. **Configuration modülü için**:
   - `config.py` veya `config.json` oluştur
   - Environment variables kullan
   - Dockerfile'da environment setup

3. **Utility modülleri için**:
   - `utils/` klasörü oluştur
   - Common functions grupla
   - Clear interface definitions
