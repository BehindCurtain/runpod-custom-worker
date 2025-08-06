# RunPod Custom Worker - Alt Sistemler

## Sistem Genel Bakış

RunPod Custom Worker projesi, 4 ana alt sistemden oluşur. Her alt sistem belirli sorumlulukları üstlenir ve diğer sistemlerle tanımlanmış arayüzler üzerinden etkileşim kurar.

## 1. İş İşleme Sistemi (Job Processing System)

### Sorumluluklar
- **Template-based** Stable Diffusion XL görüntü üretimi (Merged model yaklaşımı)
- Sınırsız uzun prompt desteği (True LPW-SDXL ile)
- Multi-template checkpoint ve LoRA yönetimi
- Template seçimine göre dinamik model yükleme
- **Build-time LoRA merging** ile optimized runtime performance
- Base64 formatında görüntü döndürme
- Hata yönetimi ve logging (fallback yok)
- Meta tensor hatalarının tamamen önlenmesi

### Ana Bileşenler
- `handler.py` - Template-aware görüntü üretim mantığı
- `template_manager.py` - Merged model pipeline yönetimi
- `templates.py` - Template konfigürasyonları
- `merge_template_loras.py` - Build-time LoRA merging sistemi
- True LPW-SDXL pipeline - Sınırsız prompt desteği
- Build-time multi-template dönüştürme sistemi - SafeTensors → Diffusers + LoRA Merge
- Diffusers pipeline yönetimi (sadece from_pretrained)
- RunPod SDK entegrasyonu

### Giriş/Çıkış
- **Giriş**: JSON formatında prompt ve parametreler
- **Çıkış**: Base64 encoded görüntü ve metadata (merged model bilgisi dahil)

### Kritik Özellikler
- GPU optimized inference
- Unlimited prompt support via True LPW-SDXL (no 77 token limit)
- **Build-time LoRA merging** - Runtime'da LoRA loading yok
- **Template-specific merged models** - Her template için optimize edilmiş checkpoint
- Volume mount shadowing protection (build-time conversion + runtime copy)
- Memory efficient processing with smart CPU offload
- **Eliminated runtime LoRA adapter management** - Daha hızlı inference
- Reproducible generation (seed control)
- Complete meta tensor error elimination
- Build-time checkpoint conversion + LoRA merging to template-specific directories
- Runtime automatic merged model detection and loading
- Optimized pipeline loading from pre-merged Diffusers format

## 2. Bağımlılık Yönetim Sistemi (Dependency Management System)

### Sorumluluklar
- AI/ML kütüphanelerinin yönetimi
- Diffusers ve PyTorch versiyonlarının kontrolü
- GPU acceleration dependencies

### Ana Bileşenler
- `requirements.txt` - AI/ML paket listesi (güncellenmiş versiyonlar)
- PyTorch 2.6.x+cu118 CUDA support
- Diffusers 0.34.x pipeline dependencies
- Accelerate 1.9.x memory optimization
- PEFT 0.17.x LoRA adapter support

### Kritik Özellikler
- CUDA 11.8.0 compatibility (PyTorch cu118 build)
- Updated package versions for better stability
- Enhanced Stable Diffusion XL support
- Improved long prompt handling capabilities
- SafeTensors format support

## 3. Container Yönetim Sistemi (Container Management System)

### Sorumluluklar
- AI/ML optimized container oluşturma
- Build-time checkpoint dönüştürme işlemi
- Network volume mount management
- Model cache directory setup
- GPU runtime optimization

### Ana Bileşenler
- `Dockerfile` - AI/ML container tanımı
- Build-time checkpoint conversion system
- Patch'li conversion script (convert_original_stable_diffusion_to_diffusers.py)
- Network volume integration
- Model storage management
- System dependencies (libgl, etc.)
- Package index strategy management

### Kritik Özellikler
- Network volume support (/runpod-volume)
- Build-time SafeTensors → Diffusers conversion with fp16 variant support
- Patch'li conversion script ile gerçek fp16 variant dosyaları oluşturma
- Model cache optimization
- GPU memory management
- Image processing libraries
- Multi-index package resolution (unsafe-best-match strategy)
- Automatic checkpoint download and conversion
- fp16 variant verification ve logging sistemi

## 4. Test ve Geliştirme Sistemi (Testing & Development System)

### Sorumluluklar
- Yerel test ortamı sağlama
- Sample input/output yönetimi
- Development workflow desteği

### Ana Bileşenler
- `test_input.json` - Test verisi
- Local testing infrastructure
- Git integration

### Kritik Özellikler
- Local execution capability
- Sample data management
- Development-production parity

## Alt Sistemler Arası İlişkiler

```
┌─────────────────────┐    ┌──────────────────────┐
│   İş İşleme         │    │   Bağımlılık         │
│   Sistemi           │◄───┤   Yönetimi           │
│                     │    │                      │
└─────────────────────┘    └──────────────────────┘
           ▲                           ▲
           │                           │
           ▼                           ▼
┌─────────────────────┐    ┌──────────────────────┐
│   Container         │    │   Test ve            │
│   Yönetimi          │◄───┤   Geliştirme         │
│                     │    │                      │
└─────────────────────┘    └──────────────────────┘
```

## Veri Akışı

1. **Development Phase**: Test Sistemi → İş İşleme Sistemi
2. **Build Phase**: 
   - Bağımlılık Sistemi → Container Sistemi
   - GitHub Script Download → Checkpoint Conversion
   - SafeTensors → Diffusers Format Conversion
3. **Runtime Phase**: Container Sistemi → İş İşleme Sistemi (Pre-converted Models)
4. **Deployment Phase**: Container Sistemi → RunPod Platform

## Genişletme Noktaları

Her alt sistem, belirli genişletme noktaları sunar:

- **İş İşleme**: Custom model loading, preprocessing/postprocessing
- **Bağımlılık**: Additional Python packages, system libraries
- **Container**: Custom base images, environment variables
- **Test**: Multiple test scenarios, integration tests
