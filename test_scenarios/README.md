# Test Senaryoları

Bu klasör, siyah görüntü sorununu izole etmek için hazırlanmış test senaryolarını içerir.

## Test Dosyaları

### 1. test_none.json
- **Amaç**: Sadece checkpoint model ile test
- **LoRA Mode**: `"none"`
- **Beklenen**: Base model'in çalışıp çalışmadığını test eder
- **Kullanım**: Eğer bu çalışırsa, sorun LoRA'larda

### 2. test_single.json
- **Amaç**: Tek LoRA ile test
- **LoRA Mode**: `"single"`
- **Yüklenen LoRA**: Sadece "Detail Tweaker XL"
- **Beklenen**: Tek LoRA'nın uyumluluğunu test eder

### 3. test_multi.json
- **Amaç**: Çoklu LoRA ile test (mevcut davranış)
- **LoRA Mode**: `"multi"`
- **Yüklenen LoRA**: Tüm 9 LoRA
- **Beklenen**: Kombinasyon sorununu tespit eder

## Test Sırası

1. **Önce test_none.json** ile başlayın
   - Eğer çalışırsa → LoRA sorunu var
   - Eğer çalışmazsa → Base model sorunu

2. **Sonra test_single.json** ile devam edin
   - Eğer çalışırsa → Çoklu LoRA kombinasyon sorunu
   - Eğer çalışmazsa → LoRA yükleme sorunu

3. **Son olarak test_multi.json** ile karşılaştırın
   - Mevcut davranışı doğrular

## RunPod'da Test Etme

Her test dosyasının içeriğini RunPod endpoint'ine gönderin:

```json
// test_none.json içeriği
{
  "input": {
    "prompt": "A beautiful portrait of a woman...",
    "lora_mode": "none"
  }
}
```

## Beklenen Loglar

### test_none.json için:
```
LoRA mode requested: none
LoRA mode: none - Skipping all LoRA loading
Pipeline loaded successfully!
```

### test_single.json için:
```
LoRA mode requested: single
LoRA mode: single - Loading only Detail Tweaker XL
Loading 1 LoRA(s)...
✓ Loaded LoRA: Detail Tweaker XL with scale 1.5
```

### test_multi.json için:
```
LoRA mode requested: multi
LoRA mode: multi - Loading all LoRAs
Loading 9 LoRA(s)...
✓ Loaded LoRA: Detail Tweaker XL with scale 1.5
...
```

## Sonuç Analizi

- **Sadece none çalışıyorsa**: LoRA sorunu
- **none ve single çalışıyorsa**: Çoklu LoRA kombinasyon sorunu
- **Hiçbiri çalışmıyorsa**: Base model veya pipeline sorunu
- **Hepsi çalışıyorsa**: Sorun başka bir yerde

## Metadata Kontrolü

Her test sonucunda metadata'da şunları kontrol edin:
- `lora_mode`: Doğru mode'un ayarlandığı
- `total_loras_loaded`: Yüklenen LoRA sayısı
- `loras_failed`: Başarısız LoRA'lar
- Image stats: `min`, `max` değerleri (siyah görüntü tespiti için)
