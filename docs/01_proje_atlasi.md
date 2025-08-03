# RunPod Custom Worker - Proje Atlası

## Proje Amacı

Bu proje, RunPod Serverless platformu için özelleştirilmiş worker'lar oluşturmak amacıyla tasarlanmış bir template projesidir. RunPod'un resmi template'lerinden fork edilmiştir ve geliştiricilerin kendi AI/ML modellerini veya özel işleme mantıklarını serverless ortamda çalıştırabilmelerini sağlar.

## Temel Felsefe

- **Basitlik**: Minimal ve anlaşılır kod yapısı
- **Esneklik**: Farklı AI/ML modelleri ve işleme mantıkları için kolayca özelleştirilebilir
- **Performans**: Model yükleme ve işleme süreçlerinin optimize edilmesi
- **Taşınabilirlik**: Docker containerization ile platform bağımsızlığı

## Mimari Yaklaşım

Proje, **event-driven serverless architecture** prensiplerine dayanır:

1. **Stateless İşleme**: Her job bağımsız olarak işlenir
2. **Container-based Deployment**: Docker ile izole edilmiş çalışma ortamı
3. **Input-Output Pattern**: JSON tabanlı giriş ve çıkış formatı
4. **Hot-start Optimization**: Model yükleme işlemlerinin handler dışında yapılması

## Teknoloji Tercihleri

### Ana Teknolojiler
- **Python 3.11**: Ana programlama dili
- **RunPod SDK**: Serverless platform entegrasyonu
- **Docker**: Containerization
- **CUDA 11.8.0**: GPU desteği

### Geliştirme Araçları
- **uv**: Hızlı Python paket yöneticisi
- **Git**: Versiyon kontrolü
- **GitHub Integration**: Otomatik deployment

## Deployment Stratejisi

### Tercih Edilen Yöntem: GitHub Integration
- Otomatik build ve deployment
- Branch-based deployment
- Continuous integration

### Alternatif Yöntem: Manual Docker Build
- Yerel build ve push
- Container registry kullanımı
- Manuel template oluşturma

## Proje Hedefleri

1. **Hızlı Başlangıç**: Geliştiricilerin minimum konfigürasyonla başlayabilmesi
2. **Kolay Özelleştirme**: Handler fonksiyonunun basit modifikasyonu
3. **Test Edilebilirlik**: Yerel test imkanları
4. **Ölçeklenebilirlik**: RunPod'un serverless altyapısından faydalanma

## Kullanım Senaryoları

- AI/ML model inference
- Görüntü işleme
- Metin analizi
- Veri dönüştürme işlemleri
- API proxy servisleri
