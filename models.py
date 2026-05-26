from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import create_transform, resolve_data_config


class ModelManager:
    """Quản lý mô hình timm và trích xuất embedding ảnh."""
    def __init__(self, device: str | None = None) -> None:
        """Khởi tạo `ModelManager` và chọn thiết bị chạy (CPU/GPU).

        Args:
            device: Tên thiết bị (ví dụ: "cuda" hoặc "cpu"). Nếu `None`, tự động chọn
                "cuda" khi có GPU, ngược lại chọn "cpu".
        """
        if device is None:
            # Chọn thiết bị: ưu tiên GPU nếu có CUDA, ngược lại dùng CPU.
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._models: Dict[str, torch.nn.Module] = {}
        self._transforms: Dict[str, object] = {}

    def _build(self, model_name: str) -> Tuple[torch.nn.Module, object]:
        """Tạo mô hình và transform tương ứng theo tên model.

        Args:
            model_name: Tên model trong `timm`.

        Returns:
            Tuple `(model, transform)` sẵn sàng cho suy luận.
        """
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        model.eval().to(self.device)
        # Lấy cấu hình tiền xử lý tương ứng với model (resize, crop, normalize)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        return model, transform

    def get(self, model_name: str) -> Tuple[torch.nn.Module, object]:
        """Lấy model và transform đã được cache hoặc tạo mới khi cần.

        Args:
            model_name: Tên model trong `timm`.

        Returns:
            Tuple `(model, transform)` tương ứng.
        """
        if model_name not in self._models:
            model, transform = self._build(model_name)
            self._models[model_name] = model
            self._transforms[model_name] = transform
        return self._models[model_name], self._transforms[model_name]

    def embed(self, model_name: str, image_rgb: np.ndarray) -> torch.Tensor:
        """Trích xuất embedding từ ảnh RGB bằng model chỉ định.

        Args:
            model_name: Tên model trong `timm`.
            image_rgb: Ảnh RGB dạng numpy array (H,W,3).

        Returns:
            Tensor embedding (torch.Tensor 1-D).
        """
        model, transform = self.get(model_name)
        return self.embed_with(model, transform, image_rgb)

    def embed_with(self, model: torch.nn.Module, transform, image_rgb: np.ndarray) -> torch.Tensor:
        """Trích xuất embedding từ ảnh RGB với model và transform đã có.

        Args:
            model: Mô hình đã load (torch.nn.Module).
            transform: Hàm/transform tiền xử lý (timm transform).
            image_rgb: Ảnh RGB dạng numpy array.

        Returns:
            Tensor embedding (torch.Tensor 1-D).
        """
        # Convert numpy RGB -> PIL Image, áp dụng transform (timm transform)
        # rồi thực hiện suy luận không gradient để lấy embedding.
        pil_img = Image.fromarray(image_rgb)
        tensor = transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = model(tensor)
        return feat.squeeze(0)

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Tính cosine similarity giữa hai embedding.

        Args:
            a: Embedding thứ nhất (torch.Tensor).
            b: Embedding thứ hai (torch.Tensor).

        Returns:
            Giá trị cosine similarity (float).
        """
        a_norm = F.normalize(a, dim=0)
        b_norm = F.normalize(b, dim=0)
        return float(F.cosine_similarity(a_norm, b_norm, dim=0).item())
