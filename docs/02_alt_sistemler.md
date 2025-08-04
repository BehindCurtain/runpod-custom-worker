# RunPod Custom Worker - Alt Sistemler

## Sistem Genel Bakış

RunPod Custom Worker projesi, 4 ana alt sistemden oluşur. Her alt sistem belirli sorumlulukları üstlenir ve diğer sistemlerle tanımlanmış arayüzler üzerinden etkileşim kurar.

## 1. İş İşleme Sistemi (Job Processing System)

### Sorumluluklar
- Stable Diffusion XL görüntü üretimi
- Prompt işleme ve validasyon (uzun prompt desteği dahil)
- LoRA kombinasyonları ile stil kontrolü
- Base64 formatında görüntü döndürme
- Hata yönetimi ve logging

### Ana Bileşenler
- `handler.py` - Ana görüntü üretim mantığı
- `long_prompt_to_embedding()` - Uzun prompt chunk blend işleme
- Diffusers pipeline yönetimi
- RunPod SDK entegrasyonu

### Giriş/Çıkış
- **Giriş**: JSON formatında prompt ve parametreler
- **Çıkış**: Base64 encoded görüntü ve metadata

### Kritik Özellikler
- GPU optimized inference
- Long prompt support (>77 tokens) with chunk blending
- Memory efficient processing
- LoRA adapter management
- Reproducible generation (seed control)
- Enhanced prompt embedding handling

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
- Network volume mount management
- Model cache directory setup
- GPU runtime optimization

### Ana Bileşenler
- `Dockerfile` - AI/ML container tanımı
- Network volume integration
- Model storage management
- System dependencies (libgl, etc.)

### Kritik Özellikler
- Network volume support (/runpod-volume)
- Model cache optimization
- GPU memory management
- Image processing libraries

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
2. **Build Phase**: Bağımlılık Sistemi → Container Sistemi
3. **Runtime Phase**: Container Sistemi → İş İşleme Sistemi
4. **Deployment Phase**: Container Sistemi → RunPod Platform

## Genişletme Noktaları

Her alt sistem, belirli genişletme noktaları sunar:

- **İş İşleme**: Custom model loading, preprocessing/postprocessing
- **Bağımlılık**: Additional Python packages, system libraries
- **Container**: Custom base images, environment variables
- **Test**: Multiple test scenarios, integration tests
