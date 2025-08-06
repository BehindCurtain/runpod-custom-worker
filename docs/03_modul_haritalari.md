# RunPod Custom Worker - Modül Haritaları

## İş İşleme Sistemi Modülleri

### handler.py Modülü

**Amaç**: Template sistemi ile True LPW-SDXL (Diffusers formatı) kullanarak Stable Diffusion XL tabanlı görüntü üretimi ve LoRA yönetimi sağlar.

**Bileşenler**:
- `handler(job)` fonksiyonu - Template-aware görüntü üretim entry point
- `get_or_load_pipeline()` - Template seçimine göre pipeline yükleme/cache
- Template seçim sistemi - Input'tan template parametresi okuma
- Template listesi endpoint - Mevcut template'leri listeleme

**Sorumluluklar**:
- Prompt ve parametrelerin işlenmesi (sınırsız uzunluk)
- Sınırsız uzun prompt desteği (True LPW-SDXL ile)
- Checkpoint'i Diffusers formatına otomatik dönüştürme
- Checkpoint ve LoRA modellerinin yönetimi
- LoRA adı sanitizasyonu ve adapter yönetimi
- Bellek optimizasyonu (akıllı CPU offload)
- FP16-safe VAE yönetimi (siyah görüntü fix'i)
- Stable Diffusion XL inference
- Base64 görüntü dönüşümü
- Error handling ve logging (fallback yok)

**Volume Mount Shadowing Koruması**:
```python
# Model configuration with shadowing protection
DIFFUSERS_DIR = os.environ.get("DIFFUSERS_DIR", "/runpod-volume/models/jib-df")
DIFFUSERS_BUILD_DIR = "/app/models/jib-df"  # Build-time location (not shadowed)

def check_diffusers_format_exists():
    """Volume mount shadowing koruması ile Diffusers formatı kontrolü"""
    diffusers_path = Path(DIFFUSERS_DIR)
    model_index_path = diffusers_path / "model_index.json"
    
    # Önce volume içinde kontrol et
    if diffusers_path.exists() and model_index_path.exists():
        print(f"✓ Diffusers format found at {DIFFUSERS_DIR}")
        return True
    
    # Build konumunda kontrol et (volume tarafından gölgelenmemiş)
    build_path = Path(DIFFUSERS_BUILD_DIR)
    build_model_index = build_path / "model_index.json"
    
    if build_path.exists() and build_model_index.exists():
        print(f"✓ Diffusers format found at build location {DIFFUSERS_BUILD_DIR}")
        print(f"Copying to volume location {DIFFUSERS_DIR}...")
        
        try:
            import shutil
            os.makedirs(diffusers_path.parent, exist_ok=True)
            shutil.copytree(str(build_path), str(diffusers_path))
            print(f"✓ Successfully copied Diffusers format")
            return True
        except Exception as copy_error:
            print(f"✗ Failed to copy: {copy_error}")
            # Fallback: build konumunu doğrudan kullan
            print(f"Using build location {DIFFUSERS_BUILD_DIR} directly")
            global DIFFUSERS_DIR
            DIFFUSERS_DIR = DIFFUSERS_BUILD_DIR
            return True
    
    return False
```

**True LPW-SDXL Sistemi (Diffusers Formatı Zorunlu)**:
```python
def load_pipeline():
    """True LPW-SDXL pipeline - SADECE Diffusers formatı"""
    # Volume mount shadowing koruması
    lora_paths = setup_models()  # check_diffusers_format_exists() çağırır
    
    # Fallback YOK - sadece Diffusers formatı
    pipe = StableDiffusionXLPipeline.from_pretrained(
        DIFFUSERS_DIR,  # Shadowing koruması sonrası güncel konum
        torch_dtype=torch.float16,
        custom_pipeline="lpw_stable_diffusion_xl",  # Sınırsız prompt
        variant="fp16",
        use_safetensors=True
    )
    print("✓ True LPW-SDXL pipeline loaded from Diffusers format")

# Handler'da sınırsız prompt kullanımı
result = pipeline(
    prompt=prompt,  # SİNIRSIZ prompt - True LPW-SDXL herhangi uzunluk
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
- `template_manager` - Template yükleme ve pipeline yönetimi
- `templates` - Template konfigürasyonları ve yardımcı fonksiyonlar
- `runpod` - Serverless platform
- `diffusers` - Stable Diffusion pipeline
- `torch` - PyTorch backend
- `PIL` - Görüntü işleme
- `numpy` - Görüntü validasyonu

### templates.py Modülü

**Amaç**: Template konfigürasyonlarını JSON dosyasından yükler ve yönetim fonksiyonlarını sağlar.

**Bileşenler**:
- `_load_templates()` - JSON dosyasından template yükleme
- `get_templates()` - Template dictionary'sini getirme
- `get_default_template()` - Varsayılan template adını getirme
- `get_template()` - Template konfigürasyonu getirme
- `get_all_unique_checkpoints()` - Tüm unique checkpoint'leri listeleme
- `get_all_unique_loras()` - Tüm unique LoRA'ları listeleme
- `list_templates()` - Template listesi ve açıklamaları
- `reload_templates()` - Template'leri yeniden yükleme (development için)

**JSON Dosya Yapısı** (`templates.json`):
```json
{
  "templates": {
    "amateur_nsfw": {
      "name": "Realistic Portrait Template",
      "description": "High-quality photorealistic portraits",
      "checkpoint": {
        "name": "Jib Mix Illustrious Realistic",
        "url": "https://civitai.com/api/download/models/1590699...",
        "filename": "jib_mix_illustrious_realistic_v2.safetensors",
        "diffusers_dir": "jib-df"
      },
      "loras": [
        {
          "name": "Detail Tweaker XL",
          "url": "https://civitai.com/api/download/models/135867...",
          "scale": 1.5,
          "filename": "detail_tweaker_xl.safetensors"
        }
      ]
    }
  },
  "default_template": "amateur_nsfw"
}
```

**Özellikler**:
- JSON-based configuration (kolay düzenleme)
- Automatic file discovery (current dir veya script dir)
- Backward compatibility (TEMPLATES ve DEFAULT_TEMPLATE değişkenleri)
- Shared model storage (aynı dosya farklı template'lerde kullanılabilir)
- Template-specific LoRA scale'leri
- Unique model detection (build optimization için)
- Template validation ve error handling
- Runtime template reloading desteği

### templates.json Dosyası

**Amaç**: Template konfigürasyonlarını JSON formatında saklar.

**Yapı**:
- `templates` - Template tanımları objesi
- `default_template` - Varsayılan template adı

**Avantajları**:
- Kod değişikliği olmadan template ekleme/düzenleme
- JSON validation ile syntax kontrolü
- Version control friendly format
- Non-technical kullanıcılar tarafından düzenlenebilir
- Build-time ve runtime'da aynı dosya kullanımı

### template_manager.py Modülü

**Amaç**: Template-based merged model pipeline yükleme ve model yönetimi sağlar.

**Bileşenler**:
- `load_template_pipeline()` - Template için merged model pipeline yükleme
- `setup_template_models()` - Template merged modellerini hazırlama
- `check_diffusers_format_exists()` - Diffusers format kontrolü
- `ensure_model_exists()` - Model indirme ve varlık kontrolü
- `ensure_all_template_loras()` - Build-time LoRA hazırlama

**Pipeline Yükleme Süreci**:
```python
def load_template_pipeline(template_name=None):
    """Template için True LPW-SDXL pipeline yükle"""
    template = get_template(template_name)
    diffusers_path, lora_paths = setup_template_models(template_name)
    
    # Diffusers formatından pipeline yükle
    pipe = StableDiffusionXLPipeline.from_pretrained(
        diffusers_path,
        torch_dtype=torch.float16,
        custom_pipeline="lpw_stable_diffusion_xl"
    )
    
    # Template-specific LoRA'ları yükle
    for lora_config in template["loras"]:
        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
    
    return pipe, template, loaded_loras, failed_loras
```

**Volume Mount Shadowing Koruması**:
- Build-time'da `/app/models/` konumunda model dönüştürme
- Runtime'da volume mount kontrolü
- Automatic copy veya fallback mekanizması

### merge_template_loras.py Modülü

**Amaç**: Build-time'da template LoRA'larını base checkpoint'lere merge eder.

**Bileşenler**:
- `merge_template_loras(template_name)` - Tek template için LoRA merge
- `merge_all_templates()` - Tüm template'ler için LoRA merge
- `check_merged_model_exists(template_name)` - Merged model varlık kontrolü
- `get_merged_model_path(template_name)` - Merged model path helper
- `ensure_lora_exists(lora_config)` - LoRA dosya varlık kontrolü

**Merge Süreci**:
```python
def merge_template_loras(template_name):
    """Template LoRA'larını base checkpoint'e merge et"""
    # Base model'i yükle
    pipe = StableDiffusionXLPipeline.from_pretrained(base_path)
    
    # Her LoRA'yı sırayla merge et
    for lora_config in template["loras"]:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_config["scale"])
        pipe.unload_lora_weights()  # Memory temizle
    
    # Merged model'i kaydet
    pipe.save_pretrained(merged_path, safe_serialization=True, variant="fp16")
```

**Dosya Yapısı**:
```
/app/models/
├── jib-df/                    # Base checkpoint (Diffusers)
├── amateur_nsfw-merged-df/    # Template merged model
└── future_template-merged-df/ # Diğer template'ler
```

**Kritik Özellikler**:
- Build-time LoRA merging (runtime'da LoRA loading yok)
- Template-specific merged model'ler
- Automatic merged model detection
- Memory efficient merging process
- GPU acceleration for faster merging
- Error handling ve fallback mekanizması

### build_all_templates.py Modülü

**Amaç**: Build-time'da tüm template'lerin modellerini hazırlar ve merge eder.

**Bileşenler**:
- `download_checkpoint()` - Checkpoint indirme
- `convert_checkpoint_to_diffusers()` - Diffusers formatına dönüştürme
- `main()` - Ana build süreci (merge dahil)

**Build Süreci**:
```python
def main():
    # Tüm unique checkpoint'leri al
    unique_checkpoints = get_all_unique_checkpoints()
    
    # Her unique checkpoint için:
    for checkpoint_config in unique_checkpoints.values():
        # İndir
        checkpoint_path = download_checkpoint(checkpoint_config)
        
        # Diffusers formatına dönüştür
        convert_checkpoint_to_diffusers(checkpoint_config, checkpoint_path)
    
    # Tüm LoRA'ları hazırla
    ensure_all_template_loras()
    
    # Template'leri merge et
    from merge_template_loras import merge_all_templates
    merged_templates = merge_all_templates()
```

**Optimizasyonlar**:
- Sadece unique modeller işlenir (duplication yok)
- Parallel processing desteği
- Error handling ve retry mekanizması
- Build verification ve logging

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
- `>=x.y.z`: True LPW-SDXL için minimum versiyon gereksinimleri
- `==x.*`: AI/ML kütüphaneleri için specific major.minor version
- `+cu118`: PyTorch için CUDA 11.8 specific build
- CUDA 11.8.0 compatibility sağlanmış

**Kritik Paketler**:
- `diffusers`: Stable Diffusion XL pipeline (>=0.34.2 - True LPW-SDXL için)
- `torch`: GPU acceleration ve model loading (v2.6.x+cu118)
- `transformers`: Text encoder ve tokenizer (v4.54.x)
- `accelerate`: Memory efficient model loading (v1.9.x)
- `peft`: LoRA adapter yükleme ve yönetimi (v0.17.x)
- `safetensors`: Güvenli model format desteği ve checkpoint dönüştürme

## Container Yönetim Sistemi Modülleri

### Dockerfile Modülü

**Amaç**: Docker container tanımını ve build sürecini yönetir.

### convert_original_stable_diffusion_to_diffusers.py Modülü

**Amaç**: Patch'li conversion script - SafeTensors checkpoint'lerini gerçek fp16 variant ile Diffusers formatına dönüştürür.

**Patch Detayları**:
```python
# Orijinal kod
pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)

# Patch'li kod (fp16 variant desteği)
pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors, variant="fp16" if args.half else None)
```

**Özellikler**:
- Hugging Face'in resmi conversion script'inin patch'li versiyonu
- `--half` parametresi kullanıldığında otomatik fp16 variant oluşturur
- Geriye dönük uyumlu (variant=None standart davranış)
- Build-time'da `/app/models/jib-df` konumunda gerçek fp16 dosyaları oluşturur

**Oluşturulan Dosyalar**:
- `unet.fp16.safetensors`
- `text_encoder.fp16.safetensors`
- `text_encoder_2.fp16.safetensors`
- `vae.fp16.safetensors`
- `model_index.json` (variant metadata ile)

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
