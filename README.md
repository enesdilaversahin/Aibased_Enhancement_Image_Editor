### 🇬🇧 \[English Version]

🇹🇷 [Türkçe Sürüm](README.tr.md)

## 📦 ImageEditor - Intelligent Image Processing App

ImageEditor is a PyQt5-based GUI application for advanced AI-powered image processing. It supports operations such as denoising, super-resolution, background removal, canvas resizing, and format conversion. It integrates powerful models such as SCUNet, SwinIR, and U²-Net.

---

### 🚀 Key Features

| Feature              | Description                                                                |
| -------------------- | -------------------------------------------------------------------------- |
| 🧠 AI Processing     | SCUNet (denoising), SwinIR (super-resolution), U²-Net (background removal) |
| 🖼️ Image Conversion | Background removal, canvas adjustment, format conversion                   |
| 🧹 Modular Workflow  | Configurable step-by-step pipeline                                         |
| 🧑‍💻 GUI            | User interface powered by PyQt5                                            |
| 🚀 GPU Acceleration  | CUDA-supported performance boost                                           |

---

## 🧠 Deep Learning Models Used

### 1. **SCUNet** – Denoising

* Used for noise removal.
* Model files: `scunet_color_25.pth`, `scunet_color_15.pth`, etc.
* Model definition: `models/network_scunet.py`
* 📄 Paper: [SCUNet: Sparsity-Controlling Unet](https://arxiv.org/abs/2107.11906)
* 🔗 Repo: [https://github.com/cszn/SCUNet](https://github.com/cszn/SCUNet)

### 2. **SwinIR** – Super-Resolution

* Used for enhancing and enlarging images (e.g., 4×).
* Model definition: `models/network_swinir.py`
* 📄 Paper: [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257)
* 🔗 Repo: [https://github.com/JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)

### 3. **U²-Net** – Background Removal

* Generates an alpha mask for transparent backgrounds.
* Model definition: `models/u2net.py`
* 📄 Paper: [U²-Net: Going Deeper with Nested U-Structure](https://arxiv.org/abs/2005.09007)
* 🔗 Repo: [https://github.com/xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)
* 🔗 [Pretrained Model Download (Google Drive)](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view)

---

## ⚙️ How It Works

1. **User Builds a Workflow:** A GUI lets you select and order processing steps.
2. **Temporary Folder Usage:** Each step writes output to a separate folder (e.g., `step_0`, `step_1`, ...).
3. **Per-Step Behavior:**

   * AI step (SCUNet/SwinIR): model is loaded onto GPU and may run in tiled mode.
   * Image step (e.g. background removal, canvas adjustment): uses PIL, NumPy, U2Net, etc.
4. **Memory Management:** CUDA cache is cleared, and Python garbage collector is invoked.
5. **Final Output is Written:** Results are saved to the output directory.

---

## 🛠️ Installation

```bash
git clone https://github.com/youruser/EgaImageEditor.git
cd EgaImageEditor
pip install -r requirements.txt
```

### Also download these models:

* [SwinIR Models](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0)

  * [SwinIR-M x4](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth)
  * [SwinIR-L x4](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth)

* [U²-Net Pretrained Weights (Google Drive)](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view)

---

### Required Python Libraries

Make sure your `torch` version matches your CUDA setup. Use [PyTorch Official Installer Guide](https://pytorch.org/get-started/locally/).

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install timm thop einops PyQt5 pynvml opencv-python
```

---

### Required CUDA & cuDNN

* CUDA: [https://developer.nvidia.com/cuda-12-6-0-download-archive](https://developer.nvidia.com/cuda-12-6-0-download-archive)
* cuDNN: [https://developer.nvidia.com/cudnn-downloads](https://developer.nvidia.com/cudnn-downloads)

---

## 🧪 Running the App

```bash
python main.py
```

The GUI will launch. Select input/output folders, configure steps, and click **Start Processing**.

---

## 📁 Folder Structure

```bash
EgaImageEditor/
├── models/
│   ├── network_scunet.py
│   ├── network_swinir.py
│   ├── u2net.py
│   └── *.pth (model weights)
├── utils/
│   ├── utils_image.py
│   └── utils_model.py
├── main.py
├── README.md
└── requirements.txt
```

---

## 📸 Screenshot

* Output Sample:

  <img src="/prepare.png" width="900px"/>

---

## 📌 Notes

* SCUNet and SwinIR operate in tiled mode for large images.
* The GUI supports dark mode on Windows.
* U²-Net generates alpha masks for background removal.

---

## 💼 License & Limitations

This project is for personal use only. Commercial use or redistribution is strictly **prohibited**.
