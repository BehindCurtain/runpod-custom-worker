# RunPod Custom Worker - Alt Sistemler

## Sistem Genel Bakış

RunPod Custom Worker projesi, 4 ana alt sistemden oluşur. Her alt sistem belirli sorumlulukları üstlenir ve diğer sistemlerle tanımlanmış arayüzler üzerinden etkileşim kurar.

## 1. İş İşleme Sistemi (Job Processing System)

### Sorumluluklar
- Gelen job'ların alınması ve işlenmesi
- Input validasyonu ve parsing
- Output formatlaması ve döndürülmesi
- Hata yönetimi ve logging

### Ana Bileşenler
- `handler.py` - Ana job işleme mantığı
- RunPod SDK entegrasyonu

### Giriş/Çıkış
- **Giriş**: JSON formatında job verisi (`event["input"]`)
- **Çıkış**: İşlenmiş sonuç (string, dict, vb.)

### Kritik Özellikler
- Stateless işleme
- Exception handling
- Performance optimization

## 2. Bağımlılık Yönetim Sistemi (Dependency Management System)

### Sorumluluklar
- Python paketlerinin tanımlanması
- Bağımlılık versiyonlarının yönetimi
- Build-time dependency resolution

### Ana Bileşenler
- `requirements.txt` - Python paket listesi
- uv paket yöneticisi entegrasyonu

### Kritik Özellikler
- Minimal bağımlılık prensibi
- Version pinning stratejisi
- Build optimization

## 3. Container Yönetim Sistemi (Container Management System)

### Sorumluluklar
- Docker image oluşturma
- Runtime environment konfigürasyonu
- System dependencies yönetimi
- Python version management

### Ana Bileşenler
- `Dockerfile` - Container tanımı
- RunPod base image entegrasyonu
- CUDA runtime konfigürasyonu

### Kritik Özellikler
- Multi-stage build capability
- GPU support (CUDA 11.8.0)
- Python 3.11 default setup
- Optimized layer caching

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
