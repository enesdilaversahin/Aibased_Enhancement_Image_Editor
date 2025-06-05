## 📦 ImageEditor - Görüntü İşleme Aracı

ImageEditor, görüntüler üzerinde gelişmiş yapay zekâ tabanlı işlemler gerçekleştiren bir PyQt5 arayüz uygulamasıdır. Görüntülerin gürültü giderme, süper çözünürlük, arka plan silme, tuval yeniden boyutlandırma ve format dönüştürme gibi işlemlerini destekler. SCUNet, SwinIR ve U²-Net gibi güçlü modelleri entegre eder.

---

### 🚀 Temel Özellikler

| Özellik                | Açıklama                                                                   |
| ---------------------- | -------------------------------------------------------------------------- |
| 🧠 AI Tabanlı İşlemler | SCUNet (denoising), SwinIR (super-resolution), U²-Net (background removal) |
| 🖼️ Görüntü Dönüşümü   | Arka plan silme, tuvali ayarlama, format dönüştürme                        |
| 🧹 Modüler İş Akışı    | Adım adım yapılandırılabilir pipeline                                      |
| 🧑‍💻 GUI              | PyQt5 tabanlı kullanıcı arayüzü                                            |
| 🧠 GPU Desteği         | CUDA destekli hızlandırma                                                  |

---

## 🧠 Kullanılan Derin Öğrenme Modelleri

### 1. **SCUNet** – Denoising

* Gürültü giderme amacıyla kullanılır.
* Model dosyaları: `scunet_color_25.pth`, `scunet_color_15.pth`, vb.
* Modelin tanımı: `models/network_scunet.py`
* 📄 Paper: [SCUNet: Sparsity-Controlling Unet](https://arxiv.org/abs/2107.11906)
* 🔗 Repo: [https://github.com/cszn/SCUNet](https://github.com/cszn/SCUNet)

### 2. **SwinIR** – Super-Resolution

* Görüntü iyileştirme ve büyütme (4x gibi).
* Modelin tanımı: `models/network_swinir.py`
* 📄 Paper: [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257)
* 🔗 Repo: [https://github.com/JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)

### 3. **U²-Net** – Background Removal

* Görselin arka planını maskeler.
* Modelin tanımı: `models/u2net.py`
* 📄 Paper: [U²-Net: Going Deeper with Nested U-Structure](https://arxiv.org/abs/2005.09007)
* 🔗 Repo: [https://github.com/xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)
* https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view 

---

## ⚙️ Çalışma Mantığı

1. **Kullanıcı İş Akışını Belirler:** GUI üzerinden sırayla uygulanacak işlemler seçilir.
2. **Geçici Klasör Kullanımı:** Her adım, geçici bir klasöre çıktı üretir (`step_0`, `step_1`, ...).
3. **Her Adımda Şu Olur:**

   * AI işlemiyse (SCUNet/SwinIR): ilgili model GPU'da yüklenir, tiled modda çalışabilir.
   * Görüntü işlem adımıysa (arka plan silme, tuval ayarı vb): U2net, PIL ve NumPy kullanılır.
4. **Bellek Temizliği:** CUDA cache temizlenir, `gc.collect()` çağrılır.
5. **Sonuçlar çıktı klasörüne yazılır.**

---

## 🛠️ Kurulum

```bash
git clone https://github.com/kullanici/EgaImageEditor.git
cd EgaImageEditor
pip install -r requirements.txt
* Aşşağıdakileri indirin
https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0
https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth
* https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view 
```


### Gereken Ekstra Kütüphaneler:

```
torh kütüphanesini cudaya uygun indireceğiz ekran kartın varsa yoksa cpu verisyonunu

https://pytorch.org/get-started/locally/ buradan seçebilirsin

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install timm thop einops PyQt5 pynvml opencv-python
```

---



### Gereken Cuda ve Cudnn:

```Uyumu seçenkleri seçip kurun
https://developer.nvidia.com/cuda-12-6-0-download-archive

https://developer.nvidia.com/cudnn-downloads
```

---


## 🧪 Test

```bash
python main.py
```

Arayüz açıldığında girdi ve çıktı klasörlerini seçin, işlemleri sıraya dizin ve "İşlem Başlat" butonuna basın.



## 📁 Klasör Yapısı

```bash
EgaImageEditor/
├── models/
│   ├── network_scunet.py
│   ├── network_swinir.py
│   ├── u2net.py
│   └── *.pth (model ağırlıkları)
├── utils/
│   ├── utils_image.py
│   └── utils_model.py
├── main.py
├── README.md
└── requirements.txt
```

---

## 📸 Ekran Görüntüsü

* Çıktılar
<img src="/prepare.png" width="900px"/> 

---

## 📌 Notlar

* SCUNet ve SwinIR modelleri tiled modda çalışar.
* GUI, karanlık mod desteği sunar.
* U2Net çıktısı doğrudan alpha kanal olarak kullanılır.

---


## 📛 Kullanım Kısıtlaması

Bu depodaki kod yalnızca kişisel amaçlarla yazılmıştır. Kodun kopyalanması, ticari projelerde kullanımı **kesinlikle yasaktır**.

