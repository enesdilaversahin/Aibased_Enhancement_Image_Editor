import shutil
import os
import cv2
from torchvision import transforms
import numpy as np
import sys
import torch
import torchvision
import signal
import multiprocessing as mp
from multiprocessing import Process, Manager, Value, Lock
from pathlib import Path
from PIL import Image
import gc
import logging
import time
from pathlib import Path
import unicodedata
import re
import ctypes
from ctypes import windll
import uuid
import pynvml
from thop import profile
import tempfile
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from einops import rearrange 
from einops.layers.torch import Rearrange, Reduce
from timm.layers import trunc_normal_, DropPath,to_2tuple
from models.network_scunet import SCUNet as SCUNet
from models.network_swinir import SwinIR as SwinIR
from models.u2net import U2NET, U2NETP
from utils import utils_image as util
from utils import utils_model
from PyQt5.QtMultimedia import QSound
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QThread, QDir, QTemporaryDir, QThreadPool,QTimer,QMutex, QMutexLocker,QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QFileDialog, QLabel,
                            QComboBox, QSpinBox, QLineEdit, QCheckBox,
                            QTabWidget, QGroupBox, QSlider, QProgressBar,
                            QMessageBox, QRadioButton, QButtonGroup, QListWidget, QAction, QListWidgetItem,QMenuBar)


def is_windows_dark_mode():
    """Windows sisteminin açık mı koyu modda mı olduğunu kontrol eder."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        return value == 0  # 0 ise karanlık mod, 1 ise açık mod
    except Exception as e:
        print(f"Windows dark mode detection failed: {e}")
        return False

# Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
torch.backends.cudnn.benchmark = True
torch.set_flush_denormal(True)

class SCUNetProcessor:
    def __init__(self, model_name='scunet_color_25', noise_level=25, tile=256, tile_overlap=32, x8=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise_level = noise_level
        self.tile = tile
        self.tile_overlap = tile_overlap
        self.x8 = x8
        self.model = self._load_model(model_name)
        
    def _load_model(self, model_name):
        model_path = f'models/{model_name}.pth'
        model = SCUNet(in_nc=3, config=[4,4,4,4,4,4,4], dim=64)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict, strict=True)
        return model.to(self.device).eval()

    def process_image(self, img_path, output_path):
        # Read and preprocess exactly like test script
        img_L = util.imread_uint(str(img_path), n_channels=3)
        img_L = util.uint2single(img_L)
        if self.noise_level > 0:
            np.random.seed(0)  # 🔴 Test scriptiyle aynı seed
            img_L += np.random.normal(0, self.noise_level/255., img_L.shape)
            img_L = np.clip(img_L, 0.0, 1.0)  # 🔴 Pixel aralığı koruma

        img_L = util.single2tensor4(img_L).to(self.device)

        # Processing with official test modes
        if self.tile:
            img_E = self.process_tiled(img_L)
        else:
            if self.x8:
                img_E = utils_model.test_mode(self.model, img_L, mode=3)
            else:
                img_E = utils_model.test_mode(self.model, img_L, mode=5)

        util.imsave(util.tensor2uint(img_E), str(output_path))
        torch.cuda.empty_cache()  # İşlem sonunda temizlik
        gc.collect()

    def process_tiled(self, img_L):
        return process_tiled(self.model, img_L, self.tile, self.tile_overlap)

class SwinIRProcessor:
    def __init__(self, task='real_sr', scale=4, model_path='models/swinir_real_sr_m_4x.pth', 
                 tile=256, tile_overlap=32, large_model=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task = task
        self.scale = scale
        self.tile = tile
        self.tile_overlap = tile_overlap
        self.window_size = 8 if task not in ['jpeg_car'] else 7
        if large_model:
            model_path = 'models/Large.pth'  # Use the correct path for the large model

        self.model = self._load_model(model_path, large_model)
        
    def _load_model(self, model_path, large_model):
        model = define_model_swinir({
            'task': self.task,
            'scale': self.scale,
            'model_path': model_path,
            'large_model': large_model,
            'training_patch_size': 128
        })
        return model.to(self.device).eval()

    def process_image(self, img_path, output_path):
        # Read and preprocess exactly like test script
        img_lq = cv2.imread(str(img_path), cv2.IMREAD_COLOR).astype(np.float32) / 255.
        h, w = img_lq.shape[:2]    
        # Padding logic from test script
        img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HWC-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)      
        h_pad = (h // self.window_size + 1) * self.window_size - h
        w_pad = (w // self.window_size + 1) * self.window_size - w
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w + w_pad]

        # Processing
        if self.tile:
            output = self.process_tiled(img_lq, h, w)
        else:
            with torch.no_grad():
                output = self.model(img_lq)

        # Postprocessing like test script
        output = output[..., :h * self.scale, :w * self.scale]
        output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
        cv2.imwrite(str(output_path), (output * 255).round().astype(np.uint8))
        torch.cuda.empty_cache()  # İşlem sonunda temizlik
        gc.collect()

    def process_tiled(self, img_lq, h, w):
        return test_tiled(self.model, img_lq, self.tile, self.tile_overlap, self.scale, self.window_size)

def process_tiled(model, img_L, tile, tile_overlap):
    """From SCUNet test script"""
    b, c, h, w = img_L.size()
    tile = min(tile, h, w)
    stride = tile - tile_overlap
    
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    
    E = torch.zeros_like(img_L)
    W = torch.zeros_like(img_L)
    
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img_L[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            with torch.no_grad():
                out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)
            
            E[..., h_idx:h_idx+tile, w_idx:w_idx+tile].add_(out_patch)
            W[..., h_idx:h_idx+tile, w_idx:w_idx+tile].add_(out_patch_mask)
            
    return E.div_(W)

def test_tiled(model, img_lq, tile, tile_overlap, scale, window_size):
    """From SwinIR test script"""
    b, c, h, w = img_lq.size()
    tile = min(tile, h, w)
    assert tile % window_size == 0, "Tile size must be multiple of window_size"
    stride = tile - tile_overlap
    
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    
    E = torch.zeros(b, c, h*scale, w*scale, device=img_lq.device)
    W = torch.zeros_like(E)
    
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            with torch.no_grad():
                out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)
            
            E[..., h_idx*scale:(h_idx+tile)*scale, 
               w_idx*scale:(w_idx+tile)*scale].add_(out_patch)
            W[..., h_idx*scale:(h_idx+tile)*scale,
               w_idx*scale:(w_idx+tile)*scale].add_(out_patch_mask)
    
    return E.div_(W)

def define_model_swinir(args_dict):
    """SwinIR modelini test scriptiyle tam uyumlu olarak tanımla"""
    class Args: pass
    args = Args()
    for k,v in args_dict.items(): setattr(args, k, v)
    
    if args.task == 'real_sr':
        if not args.large_model:
            model = SwinIR(upscale=args.scale, 
                          in_chans=3,
                          img_size=64,
                          window_size=8,
                          img_range=1.,
                          depths=[6,6,6,6,6,6],
                          embed_dim=180,
                          num_heads=[6,6,6,6,6,6],
                          mlp_ratio=2,
                          upsampler='nearest+conv',
                          resi_connection='1conv')
            param_key = 'params_ema'
        else:
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
            model = SwinIR(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
            param_key = 'params_ema'

    # Ağırlıkları test scriptiyle aynı şekilde yükle
    pretrained = torch.load(args.model_path)
    model.load_state_dict(pretrained[param_key] if param_key in pretrained else pretrained, strict=True)
    
    return model.eval()


class ProcessingConfig:
    def __init__(self, output_dir: str, target_size: tuple,
                 bg_color: str = 'white',
                 output_format: str = 'PNG',
                 output_quality: int = 90):
        self.output_dir = output_dir
        self.target_size = target_size
        self.bg_color = bg_color
        self.output_format = output_format
        self.output_quality = output_quality

class ProcessingThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, pipeline: list, input_dir: str, output_dir: str, config: ProcessingConfig, main_window=None):
        super().__init__()
        self.pipeline = pipeline
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config
        self.temp_dir = Path("C:/Temp/ImageEditor")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.main_window = main_window
        self.name_mapping = {}
        self._should_cancel = False
        self._lock = QMutex()
        self._io_lock = QMutex()

    def run(self):
        try:
            self._prepare_temp_directory()
            self._preprocess_files()

            for step_idx, step in enumerate(self.pipeline):
                if self.should_cancel:
                    logger.info("İşlem iptal edildi, çıkılıyor...")
                    break

                QApplication.processEvents()
                prev_dir = self.temp_dir / f"step_{step_idx}"
                current_dir = self.temp_dir / f"step_{step_idx + 1}"
                current_dir.mkdir(exist_ok=True)

                self._update_main_progress(step_idx)
                
                if step_idx % 100 == 0:
                    logger.info(f"[Throttle] {step_idx} görsel işlendi. Bellek temizleniyor, bekleniyor...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(10)  # Sistem nefes alsın

                try:
                    if step['type'] in ['scunet_denoise', 'swinir_process']:
                        self._process_ai_step(step, prev_dir, current_dir)
                    elif step['type'] in ['remove_background', 'adjust_canvas', 'convert_format']:
                        self._process_image_step(step, prev_dir, current_dir)

                    self._validate_step_output(current_dir)
                    QApplication.processEvents()

                except Exception as e:
                    self._handle_step_error(step_idx, e)
                    raise

            if not self.should_cancel:
                self._copy_final_results()

        except Exception as e:
            self._handle_critical_error(e)
        finally:
            self._cleanup_resources()
            if self.should_cancel and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.finished.emit()

    def _generate_safe_name(self, index: int) -> str:
        return f"{index}.png"

    def _sanitize_filename(self, filename: str) -> str:
        name = Path(filename).stem
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        return cleaned[:50]  # Removed the lowercase conversion


    def _clean_path(self, path: Path) -> Path:
        cleaned_parts = []
        for part in path.parts:
            cleaned = unicodedata.normalize('NFC', part)
            cleaned = re.sub(r'[<>:"/\\|?*]', '_', cleaned)
            cleaned_parts.append(cleaned)
        return Path(*cleaned_parts)

    def _prepare_temp_directory(self):
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        temp_step0 = self.temp_dir / "step_0"
        temp_step0.mkdir(exist_ok=True)

    def _process_ai_step(self, step: dict, input_dir: Path, output_dir: Path):
        processor = self._create_ai_processor(step)

        total_files = len(list(input_dir.glob('*')))
        processed_files = 0

        for img_path in input_dir.glob('*'):
            if self.should_cancel:
                return

            try:
                output_path = output_dir / img_path.name
                processor.process_image(img_path, output_path)
                processed_files += 1
                progress_value = int((processed_files / total_files) * 100)
                self.progress.emit(progress_value, f"{step['type']}: {img_path.name[:15]}...")

                QApplication.processEvents()

            except Exception as e:
                self._handle_file_error(img_path.name, e)

    def _process_image_step(self, step: dict, input_dir: Path, output_dir: Path):
        processor = ImageProcessor(
            operation=step['type'],
            input_dir=input_dir,
            output_dir=output_dir,
            config=self.config,
            step_config=step,
            output_format=step.get('format', 'PNG'),
            quality=step.get('quality', 90),
            width=step.get('width', self.config.target_size[0]),
            height=step.get('height', self.config.target_size[1]),
            bg_color=step.get('bg_color', self.config.bg_color)
        )
        processor.run()

    def _create_ai_processor(self, step: dict):
        tile = step.get('tile', 256)
        tile_overlap = step.get('tile_overlap', max(16, tile // 8))  # 🔥 Dinamik hesaplama
        
        if step['type'] == 'scunet_denoise':
            return SCUNetProcessor(
                model_name=step.get('model', 'scunet_color_25'),
                noise_level=step.get('noise_level', 25),
                tile=tile,
                tile_overlap=tile_overlap,
                x8=step.get('x8', False)
            )
        elif step['type'] == 'swinir_process':
            return SwinIRProcessor(
                task=step.get('task', 'real_sr'),
                scale=step.get('scale', 4),
                model_path=step.get('model_path', 'models/swinir_real_sr_m_4x.pth'),
                tile=tile,
                tile_overlap=tile_overlap,
                large_model=step.get('large_model', False)
            )


    def _validate_step_output(self, directory: Path):
        if not any(directory.iterdir()):
            self.error.emit(f"{directory.name} adımında çıktı üretilemedi!")

    def _update_main_progress(self, step_idx: int):
        progress_percent = int((step_idx / len(self.pipeline)) * 100)
        self.progress.emit(progress_percent, f"Adım {step_idx+1}/{len(self.pipeline)} işleniyor...")

    def _emit_file_progress(self, step: dict, filename: str):
        self.progress.emit(1, f"{step['type']}: {filename[:15]}...")

    def _handle_file_error(self, filename: str, error: Exception):
        error_msg = f"{filename} işlenemedi: {str(error)[:100]}"
        logger.error(error_msg)
        self.error.emit(error_msg)

    def _get_final_output_format(self) -> tuple[str, int]:
        last_conversion_step = None

        for step in reversed(self.pipeline):
            if step['type'] == 'convert_format':
                last_conversion_step = step
                break

        if last_conversion_step:
            return (
                last_conversion_step.get('format', 'png').lower(),
                last_conversion_step.get('quality', 90)
            )
        else:
            return (
                self.config.output_format.lower(),
                self.config.output_quality
            )

    def _copy_final_results(self):
        final_dir = self.temp_dir / f"step_{len(self.pipeline)}"
        output_format, quality = self._get_final_output_format()

        for temp_file in final_dir.glob('*'):
            if temp_file.is_file():
                try:
                    safe_name = temp_file.name
                    original_relative_path = self.name_mapping.get(safe_name, safe_name)

                    # Flatten the folder structure
                    output_path = self.output_dir / Path(original_relative_path).name

                    # Use the original filename's stem and the desired output format
                    output_name = output_path.with_suffix(f".{output_format.lower()}")

                    with Image.open(temp_file) as img:
                        if output_format in ["png", "webp"] and img.mode != "RGBA":
                            img = img.convert("RGBA")
                        elif output_format in ["jpg", "jpeg"] and img.mode == "RGBA":
                            img = img.convert("RGB")

                        save_args = {"format": output_format.upper(), "quality": quality}
                        img.save(output_name, **save_args)

                except Exception as e:
                    logger.error(f"Error restoring name for {temp_file.name}: {e}")

    def _cleanup_resources(self):
        with QMutexLocker(self._io_lock):
            if torch.cuda.is_available():
                for _ in range(3):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                logger.debug("GPU bellek temizleme tamamlandı")

            gc.collect()
            logger.debug("CPU bellek temizleme tamamlandı")

    def _ignore_special_files(self, src, names):
        return [n for n in names if n.startswith('.') or os.path.islink(os.path.join(src, n))]

    def _preprocess_files(self):
        self.name_mapping.clear()
        temp_step0 = self.temp_dir / "step_0"
        temp_step0.mkdir(parents=True, exist_ok=True)

        valid_extensions = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']

        def process_directory(input_dir, temp_dir):
            for item in input_dir.iterdir():
                if item.is_dir():
                    # Recursively process subdirectories
                    sub_temp_dir = temp_dir / item.name
                    sub_temp_dir.mkdir(parents=True, exist_ok=True)
                    process_directory(item, sub_temp_dir)
                elif item.is_file() and item.suffix.lower() in valid_extensions:
                    try:
                        # Generate a safe sequential name
                        relative_path = item.relative_to(self.input_dir)
                        safe_name = self._generate_safe_name_with_path(relative_path)
                        output_path = temp_dir / safe_name

                        with Image.open(item) as img:
                            if img.mode != 'RGBA':
                                img = img.convert('RGB')
                            img.save(output_path, 'PNG', optimize=True)

                        # Map the original name without converting to lowercase
                        self.name_mapping[safe_name] = str(relative_path)

                    except Exception as e:
                        logger.error(f"Preprocessing error: {item.name} → {str(e)}")

        process_directory(self.input_dir, temp_step0)
        logger.debug(f"Processed {len(self.name_mapping)} files in temp directory")

    def _generate_safe_name_with_path(self, relative_path):
        # Generate a safe name that includes the relative path
        safe_path = self._sanitize_filename(str(relative_path))
        return safe_path + ".png"
        
    @property
    def should_cancel(self):
        with QMutexLocker(self._lock):
            return self._should_cancel

    def cancel_processing(self):
        with QMutexLocker(self._lock):
            self._should_cancel = True
            logger.info("İşlem iptal ediliyor...")

        QTimer.singleShot(100, self._cleanup_after_cancel)

    def _cleanup_after_cancel(self):
        self._cleanup_resources()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.finished.emit()

    def _handle_step_error(self, step_idx: int, error: Exception):
        error_msg = f"Adım {step_idx+1} hatası: {str(error)[:200]}"
        logger.error(error_msg)
        self.error.emit(error_msg)

    def _handle_critical_error(self, error: Exception):
        error_msg = f"Kritik hata: {str(error)[:500]}"
        logger.critical(error_msg)
        self.error.emit(error_msg)
        self.finished.emit()


class ImageProcessor(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        operation: str,
        input_dir: Path,
        output_dir: Path,
        config,
        step_config: dict,
        width: int = 1080,
        height: int = 1080,
        output_format: str = "PNG", 
        quality: int = 90,
        bg_color: str = "white"
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.operation = operation
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config
        self.step_config = step_config
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.is_running = True
        self.total_files = 0  
        self.processed_files = 0  
        self.supported_formats = ['.webp', '.jpg', '.jpeg', '.png', '.bmp']
        self.output_format = output_format
        self.quality = quality
        self.u2net = None

    def run(self):
        try:
            logger.info("ImageProcessor başlatıldı: %s", self.operation)
            logger.debug("Girdi: %s, Çıktı: %s", self.input_dir, self.output_dir)

            input_dir = Path(self.input_dir).resolve()
            output_dir = Path(self.output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

            processor_fn = self._get_processor_function()
            self._process_directory(input_dir, output_dir, processor_fn)

            self.finished.emit()

        except Exception as e:
            logger.error("%s hatası: %s", self.operation, str(e), exc_info=True)
            self.error.emit(f"{self.operation} hatası: {str(e)}")

    def _get_processor_function(self):
        processors = {
            "remove_background": self._remove_background,
            "adjust_canvas": self._adjust_canvas,
            "convert_format": self._convert_format 
        }
        return processors.get(self.operation)

    def _process_directory(self, input_dir: Path, output_dir: Path, processor_fn):
        try:
            # Desteklenen formatları filtrele
            file_list = [f for f in input_dir.glob("*") if f.is_file() and f.suffix.lower() in self.supported_formats]
            self.total_files = len(file_list)
            
            for idx, img_path in enumerate(file_list, 1):
                if not self.is_running:
                    break

                try:
                    # Progress hesaplama (sıfıra bölme koruması)
                    progress = int((idx / max(1, self.total_files)) * 100)
                    self.progress.emit(progress, f"İşleniyor: {img_path.name[:15]}...")

                    # İşlemi yap ve sonucu al
                    result_img = processor_fn(img_path)
                    if result_img:
                        output_extension = self.output_format.lower()
                        output_path = output_dir / f"{img_path.stem}.{output_extension}"
                        
                        # JPEG çıktısı için alpha kanalını kontrol et
                        if self.output_format.upper() in ["JPEG", "JPG"] and result_img.mode == "RGBA":
                            result_img = result_img.convert("RGB")
                        
                        # Ortak kaydetme parametreleri
                        save_args = {"format": self.output_format.upper(), "quality": self.quality}
                        if self.output_format.upper() == "PNG":
                            save_args["optimize"] = True
                        elif self.output_format.upper() == "WEBP":
                            save_args["method"] = 6
                            save_args["quality"] = self.quality
                        
                        result_img.save(output_path, **save_args)
                except Exception as e:
                    logger.error(f"{img_path} işlenemedi: {str(e)}")
                    self.error.emit(f"Hata: {img_path.name}")

        except Exception as e:
            logger.critical(f"Klasör işleme hatası: {str(e)}")
            self.error.emit("Kritik sistem hatası!")
    
    def _convert_format(self, img_path: Path) -> Image.Image:
        try:
            output_extension = self.output_format.lower()
            output_path = self.output_dir / f"{img_path.stem}.{output_extension}"
            
            with Image.open(img_path) as img:
                # Eğer çıktı formatı PNG/WEBP ise alpha kanalını koru
                if self.output_format.upper() in ["PNG", "WEBP"]:
                    if img.mode != "RGBA":
                        img = img.convert("RGBA")
                # JPEG için alpha kanalını kaldır
                elif self.output_format.upper() in ["JPEG", "JPG"] and img.mode == "RGBA":
                    img = img.convert("RGB")
                
                save_args = {
                    "format": self.output_format.upper(),
                    "quality": self.quality
                }
                if self.output_format.upper() == "WEBP":
                    save_args["method"] = 6
                
                img.save(output_path, **save_args)
                return Image.open(output_path)
                
        except Exception as e:
            logger.error(f"Format dönüşüm hatası ({img_path.name}): {str(e)}")
            return None
                      
    def _load_u2net_model(self):
        """U²-Net modelini yükler (U2NET mimarisi)"""
        # Model dosyası "models/u2net.pth" olarak varsayılıyor
        model_path = os.path.join("models", "u2net.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    
        # Eğer self.u2net tanımlı değilse ya da None ise yükle
        if not hasattr(self, 'u2net') or self.u2net is None:
            from models import U2NET  # U2NET sınıfını doğru paketten içe aktarın
            self.u2net = U2NET(3, 1)
            state_dict = torch.load(model_path, map_location=self.device)
            self.u2net.load_state_dict(state_dict)
            self.u2net.to(self.device)
            self.u2net.eval()

    def _remove_background(self, img_path: Path) -> Image.Image:
        """U²-Net kullanarak arka plan kaldırma (transparan çıktı)"""
        try:
            # Modeli yükle (sadece bir kez)
            self._load_u2net_model()

            # Görüntüyü aç
            with Image.open(img_path).convert("RGB") as img:
                original_size = img.size  # (width, height)

                # Modelin girdi boyutu için yeniden boyutlandırma (örneğin 320x320)
                target_size = (320, 320)
                img_resized = img.resize(target_size, Image.BILINEAR)
                img_np = np.array(img_resized)
                # Normalize ve kanal sırasını (C, H, W) yap
                input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                input_tensor = input_tensor.to(self.device)

                # Modeli çalıştır
                with torch.no_grad():
                    # U²-Net çıktısı yedi ölçekli çıktı verir, d1 en büyük ayrıntıyı verir
                    d1, d2, d3, d4, d5, d6, d7 = self.u2net(input_tensor)
                    pred = d1[:, 0, :, :]  # Alfa maskesi olarak kullanılacak
                    # Normalizasyon
                    ma = torch.max(pred)
                    mi = torch.min(pred)
                    norm_pred = (pred - mi) / (ma - mi)
                    mask = (norm_pred.squeeze().cpu().numpy() * 255).astype(np.uint8)

                # Maskeyi orijinal boyuta yeniden ölçeklendir
                mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)

                # U2NET çıktı maskesi, maskede siyah kısım transparan olacak şekilde kullanılır.
                img_array = np.array(img.convert("RGBA"))
                # Alpha kanalını mask ile değiştir (mask 0 ise tamamen şeffaf)
                img_array[:, :, 3] = mask

                return Image.fromarray(img_array)

        except Exception as e:
            logger.error(f"Background removal error: {str(e)}")
            self.error.emit(f"Background removal error: {str(e)}")
            return None

    def _adjust_canvas(self, img_path: Path) -> Image.Image:
        """Adjust the canvas size and save the result"""
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGBA")
                bbox = img.getbbox()
                if not bbox:
                    return None
                img_cropped = img.crop(bbox)
                img_w, img_h = img_cropped.size
                scale = min(self.width / img_w, self.height / img_h) * 0.9
                new_size = (int(img_w * scale), int(img_h * scale))
                resized_img = img_cropped.resize(new_size, Image.LANCZOS)
                canvas = Image.new("RGBA", (self.width, self.height), self._parse_bg_color(self.bg_color))
                paste_x = (self.width - new_size[0]) // 2
                paste_y = (self.height - new_size[1]) // 2
                canvas.paste(resized_img, (paste_x, paste_y), resized_img)
                return canvas
        except Exception as e:
            logger.error(f"Canvas adjustment error: {str(e)}")
            self.error.emit(f"Canvas adjustment error: {str(e)}")
            return None

    def _parse_bg_color(self, color_str: str) -> tuple:
        color_map = {
            "white": (255, 255, 255, 255),
            "black": (0, 0, 0, 255),
            "transparent": (0, 0, 0, 0)
        }
        return color_map.get(color_str.lower(), (255, 255, 255, 255))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("MainWindow başlatılıyor...")
        self.setWindowIcon(QIcon("icon/icon.ico"))  # Ensure icon.ico exists
        self.setGeometry(100, 100, 1200, 800)
        self._update_gpu_stats()
        self._init_gpu_monitoring()
        self.setWindowTitle("ImageEditor")
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Detect Windows dark mode (implement this function if needed)
        self.is_dark_mode = self.is_windows_dark_mode()

        # Temporary directory control
        self.temp_dir = QTemporaryDir()
        if not self.temp_dir.isValid():
            QMessageBox.critical(self, "Hata", "Geçici dizin oluşturulamadı!")
            return
        self.temp_path = self.temp_dir.path()

        # Initialize default values
        self.input_folder = ""
        self.output_folder = ""
        self.pipeline_steps = []

        # Initialize ProcessingConfig
        self.config = ProcessingConfig(
            output_dir="",
            target_size=(1080, 1080),
            bg_color='white'
        )
        self.step_configs = {}

        # Setup UI components
        self.setup_ui()

        # Set initial theme
        self.set_dark_mode(self.is_dark_mode)

    def is_windows_dark_mode(self):
        # Implement the logic to detect Windows dark mode
        return False

    def apply_theme(self, dark_mode: bool):
        """Apply theme changes."""
        if dark_mode:
            self.setStyleSheet(self.dark_mode_stylesheet)
            self.title_bar.setStyleSheet("background-color: #2C2C2E;")
            self.title_label.setStyleSheet("color: white; font-size: 20px; background-color: transparent;")
        else:
            self.setStyleSheet(self.light_mode_stylesheet)
            self.title_bar.setStyleSheet("background-color: #F2F2F7;")
            self.title_label.setStyleSheet("color: black; font-size: 20px; background-color: transparent;")


    def set_dark_mode(self, enabled):
        """Set the dark mode state and update the UI accordingly."""
        self.is_dark_mode = enabled
        self.apply_theme(enabled)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.title_bar.underMouse():
            self.drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.title_bar.underMouse():
            self.move(event.globalPos() - self.drag_pos)
            event.accept()

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            self.maximize_button.setIcon(QIcon("icon/maximize.png"))
        else:
            self.showMaximized()
            self.maximize_button.setIcon(QIcon("icon/restore.png"))

    def toggle_dark_mode(self, checked):
        self.set_dark_mode(checked)

    def setup_ui(self):
        # Default style sheet
        self.light_mode_stylesheet = """
            QMainWindow { background-color: #FFFFFF; }
            QMessageBox,QInputDialog{background-color:#FFFFFF;color:#1C1C1E;}
            QPushButton { background-color: #0A84FF; color: white; border: none; padding: 10px; border-radius: 5px; }
            QPushButton:hover { background-color: #007AFF; }
            QLineEdit { padding: 5px; border: 1px solid #D1D1D6; border-radius: 5px; color: #1C1C1E; }
            QLabel { font-size: 14px; color: #1C1C1E; }
            QProgressBar { border: 1px solid #D1D1D6; border-radius: 5px; text-align: center; }
            QProgressBar::chunk { background-color: #0A84FF; width: 20px; }
            QTabWidget::pane { border: 1px solid #D1D1D6; border-radius: 5px; }
            QTabBar::tab { background: #F2F2F7; border: 1px solid #D1D1D6; padding: 10px; border-top-left-radius: 5px; border-top-right-radius: 5px; color: #1C1C1E; }
            QTabBar::tab:selected { background: #FFFFFF; border-bottom-color: #FFFFFF; }
            QListWidget { background-color: #FFFFFF; border: 1px solid #D1D1D6; color: #1C1C1E; }
            QGroupBox { border: 1px solid #D1D1D6; border-radius: 5px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; color: #1C1C1E; }
            QStatusBar{background:#F2F2F7;color:#1C1C1E;}
            QCheckBox { color: #1C1C1E; font-size: 14px; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QCheckBox::indicator:checked { background-color: #0A84FF; border-radius: 3px; }
            QCheckBox::indicator:unchecked { background-color: #D1D1D6; border-radius: 3px; }
            QComboBox {background-color: #2C2C2E; color: white; }
        """

        # Dark mode style sheet
        self.dark_mode_stylesheet = """
            QMainWindow { background-color: #1C1C1E; }
            QMessageBox, QInputDialog { background-color:#2C2C2E; color:#FFFFFF; }
            
            /* Genel Buton Stili */
            QPushButton { 
                background-color: #0A84FF; 
                color: white; 
                border: none; 
                padding: 10px; 
                border-radius: 5px; 
            }
            QPushButton:hover { background-color: #007AFF; }
            QPushButton:pressed { background-color: #0051A8; }
            QPushButton:disabled { background-color: #48484A; color: #6E6E70; }
            
            #title_bar QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 21px;
            }
            #title_bar QPushButton:hover {
                background-color: rgba(255,255,255,0.1);
            }
            #title_bar QPushButton#close_button:hover {
                background-color: #FF3B30;
            }
            
            /* Giriş Alanları */
            QLineEdit { 
                padding: 5px; 
                border: 1px solid #48484A; 
                border-radius: 5px; 
                background-color: #2C2C2E; 
                color: white; 
            }
            QLineEdit:focus { border: 2px solid #0A84FF; }
            
            /* SpinBox */
            QSpinBox {
                background-color: #2C2C2E;
                color: white;
                border: 1px solid #48484A;
                border-radius: 5px;
                padding: 5px;
            }
            QSpinBox::up-button, QSpinBox::down-button { 
                subcontrol-origin: border;
                subcontrol-position: right;
                width: 24px;
                border-left: 1px solid #48484A;
            }
            QSpinBox::up-arrow, QSpinBox::down-arrow { 
                image: url(icons/white_arrow.png);
                width: 12px;
                height: 12px;
            }
            
            /* Listeler ve Scrollbar */
            QListWidget { 
                background-color: #2C2C2E; 
                border: 1px solid #48484A; 
                color: white; 
            }
            QScrollBar:vertical {
                background: #2C2C2E;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #48484A;
                min-height: 20px;
                border-radius: 5px;
            }
            
            /* Grup Kutuları */
            QGroupBox { 
                border: 1px solid #48484A; 
                border-radius: 5px; 
                margin-top: 10px; 
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top left; 
                padding: 0 3px; 
                color: white; 
                left: 10px;
            }
            
            /* Sekmeler */
            QTabWidget::pane { border: 1px solid #48484A; border-radius: 5px; }
            QTabBar::tab { 
                background: #2C2C2E; 
                border: 1px solid #48484A; 
                padding: 12px 20px;  /* Büyük ikonlar için daha geniş */
                border-top-left-radius: 5px; 
                border-top-right-radius: 5px; 
                color: white; 
            }
            QTabBar::tab:selected { 
                background: #1C1C1E; 
                border-bottom-color: #1C1C1E; 
            }
            
            /* CheckBox & RadioButton */
            QCheckBox { color: #FFFFFF; spacing: 8px; }            
            QCheckBox::indicator { width: 18px; height: 18px; }
            QCheckBox::indicator:checked { background-color: #0A84FF; }
            QCheckBox::indicator:unchecked { background-color: #48484A; }
            
            /* ProgressBar */
            QProgressBar { 
                border: 1px solid #48484A; 
                border-radius: 5px; 
                text-align: center; 
                background-color: #2C2C2E; 
            }
            QProgressBar::chunk { 
                background-color: #0A84FF; 
                width: 20px; 
                border-radius: 3px; 
            }
            
            /* Diğer */
            QLabel { font-size: 14px; color: #FFFFFF; }
            QStatusBar { background:#2C2C2E; color:#FFFFFF; }
        """

        # Custom title bar
        self.title_bar = QWidget(self)
        self.title_bar.setFixedHeight(50)  # ✅ Üst çubuğu da büyütelim

        # Icon and Title
        self.icon_label = QLabel(self.title_bar)
        self.icon_label.setStyleSheet("background-color: transparent;")  
    
        # Load the icon
        pixmap = QPixmap("icon/icon.png")
        if pixmap.isNull():
            print("Failed to load icon.png. Please check the file path and format.")
        else:
            self.icon_label.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))  # ✅ Boyut büyüdü

        self.icon_label.setFixedSize(60, 60)  # ✅ Daha büyük alan kaplaması için genişlettik

        self.title_label = QLabel("ImageEditor", self.title_bar)
        self.title_label.setStyleSheet("color: white; font-size: 22px; background-color: transparent;")  

   
        icon_size = 28  # İstediğin boyutu buradan ayarla

        # Dark Mode Button
        self.dark_mode_button = QPushButton(self.title_bar)
        self.dark_mode_button.setFixedSize(45, 45)
        self.dark_mode_button.setCheckable(True)
        self.dark_mode_button.setChecked(self.is_dark_mode)
        self.dark_mode_button.setIcon(QIcon("icon/darkmod.png"))
        self.dark_mode_button.setIconSize(QSize(icon_size, icon_size))
        self.dark_mode_button.clicked.connect(self.toggle_dark_mode)

        # Window Controls
        self.minimize_button = QPushButton(self.title_bar)
        self.minimize_button.setFixedSize(50, 50)
        self.minimize_button.setIcon(QIcon("icon/minimize.png"))
        self.minimize_button.setIconSize(QSize(icon_size, icon_size)) 
        self.minimize_button.clicked.connect(self.showMinimized)

        self.maximize_button = QPushButton(self.title_bar)
        self.maximize_button.setFixedSize(50, 50)
        self.maximize_button.setIcon(QIcon("icon/maximize.png"))
        self.maximize_button.setIconSize(QSize(icon_size, icon_size))
        self.maximize_button.clicked.connect(self.toggle_maximize)

        self.close_button = QPushButton(self.title_bar)
        self.close_button.setFixedSize(50, 50) 
        self.close_button.setIcon(QIcon("icon/close.png"))
        self.close_button.setIconSize(QSize(icon_size, icon_size)) 
        self.close_button.clicked.connect(self.close)

        # Title Bar Layout
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(5, 0, 5, 0)
        title_layout.addWidget(self.icon_label)
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.dark_mode_button)
        title_layout.addWidget(self.minimize_button)
        title_layout.addWidget(self.maximize_button)
        title_layout.addWidget(self.close_button)
        
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 5, 10, 10)
        main_layout.setSpacing(15)
        
        # Başlık çubuğu
        main_layout.addWidget(self.title_bar)

        # Sekmeleri üst kısma yerleştir
        self.tabs = QTabWidget()
        self.tabs.setMinimumSize(1100, 550)  # Genişlik ve yükseklik artırıldı
        self.workflow_tab = QWidget()
        self.tabs.addTab(self.workflow_tab, "İş Akışı")
        self.setup_workflow_tab()
        main_layout.addWidget(self.tabs, 1)  # 1 stretch factor ile

        # Girdi/Çıktı grubunu yeniden düzenle
        self.create_io_group()
        main_layout.addWidget(self.file_group)

        # İşlem kontrolleri
        control_layout = QHBoxLayout()
        self.process_btn = QPushButton("İşlem Başlat")
        self.process_btn.setFixedSize(150, 45)
        self.cancel_btn = QPushButton("İptal")
        self.cancel_btn.setFixedSize(150, 45)
        control_layout.addStretch()
        control_layout.addWidget(self.process_btn)
        control_layout.addWidget(self.cancel_btn)
        control_layout.addStretch()

        # Progress bar ve durum
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(25)
        self.status_label = QLabel("Hazır")
        self.status_label.setAlignment(Qt.AlignCenter)

        # Layout'a ekleme sırası
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Bağlantılar
        self.connect_buttons()
        self.update_process_button_state()
        
    def connect_buttons(self):
        """Tüm buton bağlantılarını merkezi olarak yönet"""
        self.input_folder_btn.clicked.connect(self.select_input_folder)
        self.output_btn.clicked.connect(self.select_output_folder)
        self.process_btn.clicked.connect(self.start_processing)
        self.cancel_btn.clicked.connect(self.cancel_processing)

    def create_io_group(self):
        """Girdi/Çıktı grubunu oluşturur"""
        self.file_group = QGroupBox("Girdi/Çıktı Ayarları")
        layout = QVBoxLayout()
        
        # Girdi bölümü
        input_layout = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Girdi klasörü seçin...")
        self.input_folder_btn = QPushButton("Klasör Seç")
        input_layout.addWidget(QLabel("Girdi:"), 0)
        input_layout.addWidget(self.input_path_edit, 1)
        input_layout.addWidget(self.input_folder_btn, 0)

        # Çıktı bölümü
        output_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Çıktı klasörü seçin...")
        self.output_btn = QPushButton("Klasör Seç")
        output_layout.addWidget(QLabel("Çıktı:"), 0)
        output_layout.addWidget(self.output_path_edit, 1)
        output_layout.addWidget(self.output_btn, 0)

        # Format ayarları
        format_layout = QHBoxLayout()
        self.global_format_combo = QComboBox()
        self.global_format_combo.addItems(["PNG", "JPEG", "WEBP"])
        self.global_quality_spin = QSpinBox()
        self.global_quality_spin.setRange(1, 100)
        self.global_quality_spin.setValue(90)
        format_layout.addWidget(QLabel("Çıktı Formatı:"), 0)
        format_layout.addWidget(self.global_format_combo, 1)
        format_layout.addWidget(QLabel("Kalite:"), 0)
        format_layout.addWidget(self.global_quality_spin, 1)

        # Layout'a ekle
        layout.addLayout(input_layout)
        layout.addLayout(output_layout)
        layout.addLayout(format_layout)
        self.file_group.setLayout(layout)

    def setup_workflow_tab(self):
        """İş Akışı sekmesini oluşturur"""
        tab_layout = QVBoxLayout()
        
        # İşlem hattı konteynırı
        pipeline_container = QGroupBox("İşlem Hattı Yönetimi")
        pipeline_layout = QHBoxLayout()

        # Mevcut adımlar paneli
        available_steps_panel = QGroupBox("Mevcut İşlemler")
        available_layout = QVBoxLayout()
        self.available_steps_list = QListWidget()
        self.available_steps_list.addItems([
            "Bulanıklık/Gürültü Temizleme",
            "Görüntü İşleme",
            "Arka Planı Kaldır",
            "Tuvali Ayarla",
            "Formatı Dönüştür"
        ])
        add_btn = QPushButton("Ekle →")
        available_layout.addWidget(self.available_steps_list)
        available_layout.addWidget(add_btn)
        available_steps_panel.setLayout(available_layout)

        # Seçili adımlar paneli
        selected_steps_panel = QGroupBox("Seçili İşlem Sırası")
        selected_layout = QVBoxLayout()
        self.selected_steps_list = QListWidget()
        
        # Kontrol butonları
        control_buttons = QHBoxLayout()
        remove_btn = QPushButton("← Kaldır")
        up_btn = QPushButton("Yukarı Taşı")
        down_btn = QPushButton("Aşağı Taşı")
        control_buttons.addWidget(remove_btn)
        control_buttons.addWidget(up_btn)
        control_buttons.addWidget(down_btn)

        selected_layout.addWidget(self.selected_steps_list)
        selected_layout.addLayout(control_buttons)
        selected_steps_panel.setLayout(selected_layout)

        # Panel yerleşimi
        pipeline_layout.addWidget(available_steps_panel, 1)
        pipeline_layout.addWidget(selected_steps_panel, 2)
        pipeline_container.setLayout(pipeline_layout)

        # Yapılandırma paneli
        config_panel = QGroupBox("Adım Yapılandırması")
        self.config_layout = QVBoxLayout()
        self.config_layout.addWidget(QLabel("Bir adım seçerek yapılandırma yapın"))
        config_panel.setLayout(self.config_layout)

        # Ana layout
        tab_layout.addWidget(pipeline_container, 3)
        tab_layout.addWidget(config_panel, 2)
        self.workflow_tab.setLayout(tab_layout)

        # Bağlantılar
        add_btn.clicked.connect(self.add_pipeline_step)
        remove_btn.clicked.connect(self.remove_pipeline_step)
        up_btn.clicked.connect(self.move_step_up)
        down_btn.clicked.connect(self.move_step_down)
        self.selected_steps_list.model().rowsInserted.connect(self.update_process_button_state)
        self.selected_steps_list.model().rowsRemoved.connect(self.update_process_button_state)
        self.selected_steps_list.currentRowChanged.connect(self.show_step_config)
        self.scunet_model_combo = QComboBox()
        self.scunet_noise_spin = QSpinBox()
        self.scunet_tile_spin = QSpinBox()
        self.x8_check = QCheckBox()
        self.swinir_task_combo = QComboBox()
        self.swinir_scale_spin = QSpinBox()
        self.swinir_model_path_edit = QLineEdit()
        self.swinir_tile_spin = QSpinBox()
        self.large_model_check = QCheckBox()
        self.canvas_width_spin = QSpinBox()
        self.canvas_height_spin = QSpinBox()
        self.bg_color_combo = QComboBox()
        
      
    def select_input_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Girdi Klasörünü Seç")
        if folder_path:
            self.input_folder = str(Path(folder_path))
            self.input_path_edit.setText(self.input_folder)
            self.update_process_button_state()  # Güncelleme tetiklendi
            self.validate_paths()  # Ekstra path doğrulama

    def select_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Çıktı Klasörü Seç")
        if folder_path:
            self.config.output_dir = str(Path(folder_path))
            self.output_path_edit.setText(self.config.output_dir)
            self.update_process_button_state()  # Güncelleme tetiklendi
            self.validate_paths()  # Ekstra path doğrulama
            
            
    def validate_paths(self):
        """Path'leri bağımsız olarak kontrol et"""
        # Input path kontrolü
        input_valid = False
        if self.input_folder:
            input_valid = Path(self.input_folder).exists()
        
        # Output path kontrolü
        output_valid = False
        if self.config.output_dir:
            output_valid = Path(self.config.output_dir).exists()
        
        # Stilleri ayrı ayrı güncelle
        self.input_path_edit.setStyleSheet(
            "border: 2px solid green" if input_valid else "border: 2px solid red"
        )
        self.output_path_edit.setStyleSheet(
            "border: 2px solid green" if output_valid else "border: 2px solid red"
        )

    def select_model_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Model Dosyasını Seç",
            "",
            "Model Dosyaları (*.pth *.pt)"
        )
        if file_path:
            self.swinir_model_path_edit.setText(file_path)

    def update_process_button_state(self):
        has_input = bool(self.input_path_edit.text())
        has_output = bool(self.output_path_edit.text())
        has_steps = self.selected_steps_list.count() > 0
        
        print(f"[DEBUG] Input: {has_input} | Output: {has_output} | Steps: {has_steps}")  # Debug
        
        self.process_btn.setEnabled(has_input and has_output and has_steps)
    
    def _init_gpu_monitoring(self):
        try:
            
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._update_gpu_stats)
            self.timer.start(1000)
        except ImportError:
            logger.warning("GPU monitoring disabled (pynvml not installed)")

    def _update_gpu_stats(self):
        try:
            if pynvml and torch.cuda.is_available():
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
                self.statusBar().showMessage(
                    f"GPU Memory: {mem_info.free//1024**2}MB free/{mem_info.total//1024**2}MB total | "
                    f"Utilization: {util.gpu}%"
                )
                pynvml.nvmlShutdown()
            elif not pynvml:
                self.statusBar().showMessage("GPU monitoring requires pynvml (pip install pynvml)")
        except Exception as e:
            logger.error(f"GPU stats error: {str(e)}"),
  
                    
    def add_pipeline_step(self):
        current_item = self.available_steps_list.currentItem()
        if current_item:
            step_text = current_item.text()
            step_uuid = str(uuid.uuid4())  # Generate UUID
            print(f"[DEBUG] Yeni Adım: {step_text} | UUID: {step_uuid}")
            item = QListWidgetItem(step_text)
            item.setData(Qt.UserRole, step_uuid)  # Store UUID in item
            self.selected_steps_list.addItem(item)
            self.update_process_button_state()
            
            # Initialize default configuration based on step type
            default_config = {}
            if step_text == "Bulanıklık/Gürültü Temizleme":
                default_config = {
                    'model_combo': 'scunet_color_25',
                    'noise_spin': 25,
                    'tile_spin': 128,
                    'x8_check': False    
                }
            elif step_text == "Görüntü İşleme":
                default_config = {
                    'task_combo': 'real_sr',
                    'scale_spin': 4,
                    'model_path_edit': 'models/swinir_real_sr_m_4x.pth',
                    'tile_spin': 128,
                    'large_model': False
                }
            elif step_text == "Tuvali Ayarla":
                default_config = {
                    'width': 1080,
                    'height': 1080,
                    'bg_color': 'white'
                }
            elif "Formatı Dönüştür" in step_text:
                  default_config = {
                        'format': 'PNG',
                        'quality': 90
                  }
            # Arka Planı Kaldır has no configuration
            self.step_configs[step_uuid] = default_config
            self.update_process_button_state()

    def remove_pipeline_step(self):
        current_row = self.selected_steps_list.currentRow()
        if current_row >= 0:
            item = self.selected_steps_list.takeItem(current_row)
            step_uuid = item.data(Qt.UserRole)
            if step_uuid in self.step_configs:
                del self.step_configs[step_uuid]
            self.update_process_button_state()
            if self.selected_steps_list.count() == 0:
                self.clear_step_config()

    def move_step_up(self):
        current_row = self.selected_steps_list.currentRow()
        if current_row > 0:
            item = self.selected_steps_list.takeItem(current_row)
            self.selected_steps_list.insertItem(current_row - 1, item)
            self.selected_steps_list.setCurrentRow(current_row - 1)

    def move_step_down(self):
        current_row = self.selected_steps_list.currentRow()
        if current_row < self.selected_steps_list.count() - 1 and current_row >= 0:
            item = self.selected_steps_list.takeItem(current_row)
            self.selected_steps_list.insertItem(current_row + 1, item)
            self.selected_steps_list.setCurrentRow(current_row + 1)

    def clear_step_config(self):
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    clear_layout(item.layout())

        for i in reversed(range(self.config_layout.count())):
            item = self.config_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                clear_layout(item.layout())

    def show_step_config(self, row):
        if row < 0:
            return
        self.clear_step_config()
        item = self.selected_steps_list.item(row)
        step_text = item.text()
        step_uuid = item.data(Qt.UserRole)  # Retrieve UUID
        
        if "Bulanıklık/Gürültü Temizleme" in step_text:
            self.setup_scunet_config(step_uuid)
        elif "Görüntü İşleme" in step_text:
            self.setup_swinir_config(step_uuid)
        elif "Tuvali Ayarla" in step_text:
            self.setup_canvas_config(step_uuid)
        elif "Arka Planı Kaldır" in step_text:
            self.setup_remove_background_config(step_uuid)
        elif "Formatı Dönüştür" in step_text:
            self.setup_format_config(step_uuid)


    def setup_remove_background_config(self, step_id):
        self.clear_step_config()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Arka Planı Kaldır: Ek yapılandırma gerekmez."))
        self.config_layout.addLayout(layout)

    def setup_scunet_config(self, step_id):
        self.clear_step_config()
        layout = QVBoxLayout()

        # Create new widget instances for SCUNet configuration
        scunet_model_combo = QComboBox()
        scunet_model_combo.addItems(['scunet_color_25','scunet_color_15', 'scunet_color_50'])
        model_layout = self.create_labeled_widget("SCUNet Model:", scunet_model_combo)

        scunet_noise_spin = QSpinBox()
        scunet_noise_spin.setRange(1, 100)
        scunet_noise_spin.setValue(25)
        noise_layout = self.create_labeled_widget("Gürültü Seviyesi:", scunet_noise_spin)
        scunet_tile_spin = QSpinBox()
        scunet_tile_spin.setRange(0, 1024)
        scunet_tile_spin.setValue(128)
        tile_layout = self.create_labeled_widget("Tile Boyutu:", scunet_tile_spin)

        x8_check = QCheckBox("x8 Self-Ensemble (Yavaş ama Kaliteli)")
        x8_check.stateChanged.connect(lambda state: self.save_step_config(
            step_id, x8_check=x8_check.isChecked()
        ))
        # Add layouts and widget to the main configuration layout
        layout.addLayout(model_layout)
        layout.addLayout(noise_layout)
        layout.addWidget(x8_check)
        layout.addLayout(tile_layout)

        # Load and connect step configuration using new widget instances
        self.load_step_config(step_id, model_combo=scunet_model_combo, noise_spin=scunet_noise_spin, tile_spin=scunet_tile_spin ,x8_check=x8_check)
        self.connect_step_config_signals(step_id, model_combo=scunet_model_combo, noise_spin=scunet_noise_spin, tile_spin=scunet_tile_spin ,x8_check=x8_check)
        self.config_layout.addLayout(layout)
                
    def setup_format_config(self, step_id):
        """Format dönüşümü için UI bileşenleri."""
        self.clear_step_config()
        layout = QVBoxLayout()
        
        # Format seçimi
        format_combo = QComboBox()
        format_combo.addItems(["PNG", "JPEG", "WEBP"])
        format_combo.setCurrentText("PNG")
        layout.addLayout(self.create_labeled_widget("Çıktı Formatı:", format_combo))
        
        # Kalite ayarı (JPEG/WEBP için)
        quality_spin = QSpinBox()
        quality_spin.setRange(1, 100)
        quality_spin.setValue(90)
        layout.addLayout(self.create_labeled_widget("Kalite (%):", quality_spin))
        
        # Config yükleme ve sinyal bağlantıları
        self.load_step_config(step_id, format_combo=format_combo, quality_spin=quality_spin)   
        self.connect_step_config_signals(step_id, format_combo=format_combo, quality_spin=quality_spin)
        self.config_layout.addLayout(layout)    

    def create_labeled_combo_box(self, label_text, items, combo_box):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        combo_box.addItems(items)
        layout.addWidget(combo_box)
        return layout
        
    def create_labeled_widget(self, label_text, widget):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        layout.addWidget(widget)
        return layout


    def create_labeled_spin_box(self, label_text, min_value, max_value, default_value, spin_box):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        spin_box.setRange(min_value, max_value)
        spin_box.setValue(default_value)
        layout.addWidget(spin_box)
        return layout

    def load_step_config(self, step_id, **widgets):
        if step_id in self.step_configs:
            for widget_name, widget in widgets.items():
                if widget_name in self.step_configs[step_id]:
                    if isinstance(widget, QComboBox):
                        widget.setCurrentText(self.step_configs[step_id][widget_name])
                    elif isinstance(widget, QSpinBox):
                        widget.setValue(self.step_configs[step_id][widget_name])
                    elif isinstance(widget, QCheckBox):
                        widget.setChecked(self.step_configs[step_id][widget_name])
                    elif isinstance(widget, QLineEdit):
                        widget.setText(self.step_configs[step_id][widget_name])

    def connect_step_config_signals(self, step_id, **widgets):
        for widget_name, widget in widgets.items():
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(
                    lambda text, widget_name=widget_name, step_id=step_id: self.save_step_config(step_id, **{widget_name: text})
                )
            elif isinstance(widget, QSpinBox):
                widget.valueChanged.connect(
                    lambda value, widget_name=widget_name, step_id=step_id: self.save_step_config(step_id, **{widget_name: value})
                )
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(
                    lambda state, widget_name=widget_name, step_id=step_id, widget=widget: self.save_step_config(step_id, **{widget_name: widget.isChecked()})
                )
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(
                    lambda text, widget_name=widget_name, step_id=step_id: self.save_step_config(step_id, **{widget_name: text})
                )

    def save_step_config(self, step_id, **config):
        if step_id not in self.step_configs:
            self.step_configs[step_id] = {}
        self.step_configs[step_id].update(config)

    def setup_swinir_config(self, step_id):
        self.clear_step_config()
        layout = QVBoxLayout()

        # Create new widget instances for SwinIR configuration
        swinir_task_combo = QComboBox()
        swinir_task_combo.addItems(['real_sr', 'lightweight_sr'])
        task_layout = self.create_labeled_widget("Görev:", swinir_task_combo)

        swinir_scale_spin = QSpinBox()
        swinir_scale_spin.setRange(1, 8)
        swinir_scale_spin.setValue(4)
        scale_layout = self.create_labeled_widget("Ölçek Faktörü:", swinir_scale_spin)

        swinir_model_path_edit = QLineEdit()
        swinir_model_path_edit.setText('models/swinir_real_sr_m_4x.pth')
        model_path_layout = self.create_labeled_widget("Model Yolu:", swinir_model_path_edit)
        model_path_btn = QPushButton("Seç")
        model_path_btn.clicked.connect(self.select_model_path)
        model_path_layout.addWidget(model_path_btn)

        swinir_tile_spin = QSpinBox()
        swinir_tile_spin.setRange(0, 1024)
        swinir_tile_spin.setValue(256)
        tile_layout = self.create_labeled_widget("Tile Boyutu:", swinir_tile_spin)

        large_model_check = QCheckBox("Large Model (Büyük Model)")

        layout.addLayout(task_layout)
        layout.addLayout(scale_layout)
        layout.addLayout(model_path_layout)
        layout.addLayout(tile_layout)
        layout.addWidget(large_model_check)

        self.load_step_config(step_id, task_combo=swinir_task_combo, scale_spin=swinir_scale_spin, model_path_edit=swinir_model_path_edit, tile_spin=swinir_tile_spin, large_model=large_model_check)
        self.connect_step_config_signals(step_id, task_combo=swinir_task_combo, scale_spin=swinir_scale_spin, model_path_edit=swinir_model_path_edit, tile_spin=swinir_tile_spin, large_model=large_model_check)

        self.config_layout.addLayout(layout)
  
    def setup_canvas_config(self, step_id):
        self.clear_step_config()
        layout = QVBoxLayout()

        # Create new widget instances for canvas configuration
        canvas_width_spin = QSpinBox()
        canvas_width_spin.setRange(0, 10000)
        size_layout = self.create_labeled_widget("Tuval Genişliği:", canvas_width_spin)

        canvas_height_spin = QSpinBox()
        canvas_height_spin.setRange(0, 10000)
        height_layout = self.create_labeled_widget("Tuval Yüksekliği:", canvas_height_spin)

        size_container = QHBoxLayout()
        size_container.addLayout(size_layout)
        size_container.addLayout(height_layout)
        layout.addLayout(size_container)

        bg_color_combo = QComboBox()
        bg_color_combo.addItems(['white', 'black', 'transparent'])
        bg_color_layout = self.create_labeled_widget("Arka Plan Rengi:", bg_color_combo)
        layout.addLayout(bg_color_layout)

        self.load_step_config(step_id, width=canvas_width_spin, height=canvas_height_spin, bg_color=bg_color_combo)
        self.connect_step_config_signals(step_id, width=canvas_width_spin, height=canvas_height_spin, bg_color=bg_color_combo)  
        self.config_layout.addLayout(layout)

    def collect_pipeline_steps(self):
        steps = []
        print("[DEBUG] Toplanan Adımlar:")
        for i in range(self.selected_steps_list.count()):
            item = self.selected_steps_list.item(i)
            step_text = item.text()
            step_uuid = item.data(Qt.UserRole)
            step_config = self.step_configs.get(step_uuid, {})
            self.update_process_button_state()
            
            print(f" - Adım {i+1}: {step_text} | UUID: {step_uuid} | Config: {step_config}")
            
            if "Bulanıklık/Gürültü Temizleme" in step_text:
                steps.append({
                    'type': 'scunet_denoise',
                    'model': step_config.get('model_combo', 'scunet_color_25'),
                    'noise_level': step_config.get('noise_spin', 25),
                    'tile': step_config.get('tile_spin', 256),
                    'x8': step_config.get('x8_check', False)

                })
            elif "Görüntü İşleme" in step_text:
                steps.append({
                    'type': 'swinir_process',
                    'task': step_config.get('task_combo', 'real_sr'),
                    'scale': step_config.get('scale_spin', 4),
                    'model_path': step_config.get('model_path_edit', 'models/swinir_real_sr_m_4x.pth'),
                    'tile': step_config.get('tile_spin', 256),
                    'large_model': step_config.get('large_model', False)
                })
            elif "Arka Planı Kaldır" in step_text:
                steps.append({'type': 'remove_background'})
            elif "Tuvali Ayarla" in step_text:
                steps.append({
                    'type': 'adjust_canvas',
                    'width': step_config.get('width', 1080),
                    'height': step_config.get('height', 1080),
                    'bg_color': step_config.get('bg_color', 'white')
                })
            elif "Formatı Dönüştür" in step_text:
                steps.append({
                    'type': 'convert_format',
                    'format': step_config.get('format_combo', 'PNG'),
                    'quality': step_config.get('quality', 90)
                })
        return steps

    def find_latest_output(self, output_dir):
        output_images = list(Path(output_dir).glob('*'))
        if output_images:
            return str(output_images[0])
        return None

    def validate_configurations(self):
        missing_configs = []
    
        # Path'leri kontrol et
        input_path = Path(self.input_folder)
        output_path = Path(self.config.output_dir)
        
        print(f"[VALIDATION] Input Path Exists: {input_path.exists()}")
        print(f"[VALIDATION] Output Path Writable: {output_path.is_dir()}")
        # 🚨 Check input and output folders
        if not self.input_folder or self.input_folder.strip() == "":
            print("DEBUG: Girdi klasörü eksik!")  # Should appear in terminal
            missing_configs.append("Girdi klasörü seçilmedi!")

        if not self.config.output_dir or self.config.output_dir.strip() == "":
            print("DEBUG: Çıktı klasörü eksik!")  # Should appear in terminal
            missing_configs.append("Çıktı klasörü seçilmedi!")

        # 🚀 Print what is missing
        print("Eksik Ayarlar:", missing_configs)

        if missing_configs:
            error_message = "\n".join(missing_configs)

            # Create the QMessageBox to display the errors
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Eksik Ayarlar")
            msg_box.setText("Aşağıdaki ayarlar eksik veya yanlış yapılandırılmış:")
            msg_box.setInformativeText(error_message)  # Display all error messages here
            msg_box.setStyleSheet("QMessageBox { background-color: white; color: black; } QLabel { color: black; }")
        
            # Play the error notification sound
            try:
                QSound.play("alert.wav")  # Make sure path is correct
            except Exception as e:
                logger.warning("Ses uyarısı çalınamadı: %s", str(e))

            msg_box.exec_()  

            return False

        print("DEBUG: Tüm ayarlar tamam.")  
        return True

    def start_processing(self):
        print("DEBUG: İşlem Başlat butonuna basıldı.") 

        # ✅ Eski bir thread varsa temizle
        if hasattr(self, 'processing_thread'):
            if self.processing_thread is not None and self.processing_thread.isRunning():
                QMessageBox.warning(self, "Hata", "Bir işlem zaten devam ediyor. Lütfen bekleyin veya işlemi iptal edin.")
                return
            self.processing_thread = None  # ✅ Thread'i sıfırla

        # ✅ Girdi / Çıktı kontrolleri
        is_valid = self.validate_configurations()
        print(f"DEBUG: Validation sonucu: {is_valid}")  

        if not is_valid:
            print("DEBUG: Eksik ayarlar bulundu, işlem başlamadı.")  
            return  

        print("DEBUG: Tüm ayarlar doğru, işlem başlatılıyor...")  

        # ✅ Yeni işlem başlat
        steps = self.collect_pipeline_steps()
        if not steps:
            QMessageBox.warning(self, "Hata", "İşlem adımları yapılandırılmadı! Lütfen işlem adımlarını belirleyin.")
            return
            
        self.config.output_format = self.global_format_combo.currentText().lower()
        self.config.output_quality = self.global_quality_spin.value()    

        self.processing_thread = ProcessingThread(
            pipeline=steps,
            input_dir=self.input_folder,
            output_dir=self.config.output_dir,
            config=self.config,
            main_window=self 
        )   

        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error.connect(self.processing_error)

        self.processing_thread.start()

        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)


    def update_progress(self, percent: int, message: str):
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def processing_finished(self):
        self.status_label.setText("İşlem başarıyla tamamlandı!")
        QMessageBox.information(self, "İşlem Tamamlandı", "İşlem başarıyla tamamlandı!")
    
        # Cleanup before resetting UI
        if hasattr(self, 'processing_thread') and self.processing_thread is not None:
            try:
                if self.processing_thread.isRunning():
                    self.processing_thread.quit()
                    self.processing_thread.wait()
                self.processing_thread.deleteLater()
            except RuntimeError:
                pass  # Handle cases where thread is already deleted
            finally:
                self.processing_thread = None  # Clear reference
    
        self.reset_processing_ui()
    def processing_error(self, error_message):
        """Sadece kritik hataları göster"""
        if "Subprocess çıktı kuyruğu boş" in error_message:
            return  # İptal kaynaklı hataları gizle
    
        QMessageBox.critical(
            self,
            "Kritik Hata",
            f"Teknik hata oluştu:\n{error_message[:300]}..."
        )

    def cancel_processing(self):
        logger.info("Initiating cancellation protocol")
        try:
            if hasattr(self, 'processing_thread') and self.processing_thread.isRunning():
                # Send cancellation signal
                self.processing_thread.cancel_processing()
            
                # Force termination if not responding
                if not self.processing_thread.wait(2000):  # 2 second grace period
                    logger.warning("Forcing thread termination")
                    self.processing_thread.terminate()
                
                self.processing_thread = None
                logger.info("Processing thread terminated")
            
            # Additional GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                logger.debug("Main thread CUDA cache cleared")
            
        except Exception as e:
            logger.error(f"Cancellation error: {str(e)}")
        finally:
            # Immediate UI update
            self.process_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.status_label.setText("İşlem anında iptal edildi")
            self.progress_bar.setValue(0)
            logger.info("UI resources reset")
    

    def reset_processing_ui(self):
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Hazır")
    
        # Only attempt deletion if thread exists
        if hasattr(self, 'processing_thread') and self.processing_thread is not None:
            try:
                self.processing_thread.deleteLater()
            except RuntimeError:
                pass
        self.processing_thread = None  # Ensure reference is cleared

    def generate_unique_step_id(self):
        import uuid
        return str(uuid.uuid4())

    def get_step_id_from_row(self, row):
        step_text = self.selected_steps_list.item(row).text()
        return step_text.split("(ID:")[-1].split(")")[0].strip()

def check_requirements():
    missing_requirements = []
    warning_messages = []  # İsim değişikliği yapıldı

    # 1. Görsel kütüphanelerin kontrolü
    try:
        from PIL import Image
        print(f"Pillow (PIL) {Image.__version__} yüklü")
    except ImportError:
        missing_requirements.append("Pillow (PIL)")
    
    try:
        import cv2
        print(f"OpenCV {cv2.__version__} yüklü")
    except ImportError:
        missing_requirements.append("OpenCV (opencv-python)")

    # 2. Torch ile ilgili kontroller
    try:
        import torch
        if not torch.cuda.is_available():
            warning_messages.append("CUDA bulunamadı (CPU modunda çalışacak)")
        else:
            print(f"Torch CUDA {torch.version.cuda} aktif")
            
    except ImportError:
        missing_requirements.append("PyTorch (torch)")

    # 3. Diğer zorunlu paketler
    required_packages = {
        'torch': 'torch',
        'PyQt5': 'PyQt5',
        'numpy': 'numpy',
        'timm': 'timm'
    }

    for display_name, package_name in required_packages.items():
        try:
            __import__(package_name)
            print(f"{display_name} yüklü")
        except ImportError:
            missing_requirements.append(display_name)

    # 4. Uyarıları filtrele
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

    # 5. Hata mesajını oluştur
    if missing_requirements:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Eksik Bağımlılıklar")
        msg.setText("Aşağıdaki paketler eksik:\n\n- " + "\n- ".join(missing_requirements))
        msg.addButton(QMessageBox.Ok)
        msg.exec_()
        return False

    if warning_messages:  # Düzeltilmiş isim
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Sistem Uyarıları")
        msg.setText("\n".join(warning_messages))  # Artık iterable bir liste
        msg.addButton(QMessageBox.Ok)
        msg.exec_()

    return True

if __name__ == '__main__':
    # Multiprocessing ve DPI ayarları
    if sys.platform.startswith('win'):
        from ctypes import windll
        windll.shell32.SetCurrentProcessExplicitAppUserModelID("YourApp.Unique.ID")
        if hasattr(windll.shcore, 'SetProcessDpiAwareness'):
            windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI
    
    # Bellek optimizasyonu
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Uygulama başlatma
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # EXE'de daha tutarlı görünüm
    
    if check_requirements():
        window = MainWindow()
        window.set_dark_mode(True) 
        window.show()
        sys.exit(app.exec_())
    else:
        sys.exit(1)