# RunPod Custom Worker - Modül Haritaları

## İş İşleme Sistemi Modülleri

### handler.py Modülü

**Amaç**: Ana job işleme mantığını içerir ve RunPod serverless entegrasyonunu sağlar.

**Bileşenler**:
- `handler(job)` fonksiyonu - Ana entry point
- RunPod SDK entegrasyonu
- Model loading alanı (global scope)

**Sorumluluklar**:
- Job input parsing (`job["input"]`)
- İş mantığının uygulanması
- Response formatlaması
- Error handling

**Bağımlılıklar**:
- `runpod` paketi
- Custom model libraries (opsiyonel)

**Genişletme Noktaları**:
```python
# Model loading (global scope)
# model = load_your_model()

def handler(job):
    # Input processing
    job_input = job["input"]
    
    # Custom logic here
    # result = model.predict(job_input)
    
    # Output formatting
    return result
```

## Bağımlılık Yönetim Sistemi Modülleri

### requirements.txt Modülü

**Amaç**: Python paket bağımlılıklarını tanımlar.

**Yapı**:
```
# Core dependency
runpod~=1.7.9

# Custom dependencies (to be added)
# torch>=2.0.0
# transformers>=4.30.0
# pillow>=9.0.0
```

**Versioning Stratejisi**:
- `~=`: Compatible release (patch level changes)
- `>=`: Minimum version requirement
- `==`: Exact version pinning (production için önerilen)

**Genişletme Rehberi**:
- AI/ML: `torch`, `tensorflow`, `transformers`
- Image Processing: `pillow`, `opencv-python`
- Data Processing: `pandas`, `numpy`
- API: `requests`, `httpx`

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
   RUN uv pip install --upgrade -r /requirements.txt
   ```
   - Package installation
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
