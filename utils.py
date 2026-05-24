from __future__ import print_function, division
import matplotlib.pyplot as plt
import math
from sklearn.metrics import auc
import numpy as np
import cv2
import os, sys
from torchvision import models, transforms
from seaborn import color_palette
import torch
import torch.nn.functional as F
from typing import List, Tuple, Any, Dict, Optional, Union
from pathlib import Path
import copy
from torch import nn

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, template_dir_path: Union[str, Path], image_name: str, thresh: float = 0.7, template_scale: float = 1.0, transform: Optional[transforms.Compose] = None) -> None:
        """
        Khởi tạo tập dữ liệu cho một ảnh mẫu và nhiều ảnh template.
        
        Args:
            template_dir_path (Union[str, Path]): Đường dẫn đến thư mục chứa các ảnh template.
            image_name (str): Đường dẫn đến ảnh mẫu (sample).
            thresh (float): Ngưỡng độ tin cậy mặc định cho các template.
            template_scale (float): Tỉ lệ thu phóng ảnh template.
            transform (transforms.Compose, optional): Các phép biến đổi áp dụng lên ảnh.
        """
        # Phép biến đổi mặc định: đổi ảnh sang tensor và chuẩn hóa theo phân bố ImageNet.
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        self.angles = [0, 45, 90, 135, 180, 225, 270, 315]
        self.base_template_paths = list(Path(template_dir_path).iterdir())
        self.template_paths = []
        for path in self.base_template_paths:
            for _ in self.angles:
                self.template_paths.append(path)

        self.image_name = image_name
        self.thresh = thresh
        self.template_scale = template_scale

        # Đọc ảnh sample một lần duy nhất và giữ lại bản thô (raw) để vẽ kết quả về sau.
        self.image_raw = cv2.imread(self.image_name)
        if self.image_raw is None:
            raise FileNotFoundError(f'Cannot read sample image: {self.image_name}')

        self.image = self.transform(self.image_raw).unsqueeze(0)

    def __len__(self) -> int:
        return len(self.template_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Lấy một phần tử (template) trong thư mục template."""
        # Mỗi phần tử tương ứng với 1 template trong thư mục.
        template_path = str(self.template_paths[idx])
        angle = self.angles[idx % len(self.angles)]

        template = cv2.imread(template_path)
        if template is None:
            raise FileNotFoundError(f'Cannot read template image: {template_path}')

        if angle != 0:
            (h, w) = template.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY
            template = cv2.warpAffine(template, M, (nW, nH), borderValue=(255, 255, 255))

        if self.template_scale != 1.0:
            h, w = template.shape[:2]
            new_w = max(1, int(w * self.template_scale))
            new_h = max(1, int(h * self.template_scale))
            interpolation = cv2.INTER_AREA if self.template_scale < 1.0 else cv2.INTER_CUBIC
            template = cv2.resize(template, (new_w, new_h), interpolation=interpolation)

        template = self.transform(template)

        return {
            'image': self.image,
            'image_raw': self.image_raw,
            'image_name': self.image_name,
            'template': template.unsqueeze(0),
            'template_name': template_path,
            'template_h': template.size()[-2],
            'template_w': template.size()[-1],
            'thresh': self.thresh,
        }


def build_feature_extractor(model_name: str = 'convnext_tiny', pretrained: bool = True) -> torch.nn.Module:
    """
    Tạo backbone trích xuất đặc trưng với nhiều lựa chọn mô hình.
    Tự động cắt bớt các tầng sâu để giữ độ phân giải đặc trưng ở mức H/8, W/8.
    
    Args:
        model_name (str): Tên mô hình ('convnext_tiny', 'efficientnet_b4', 'mobilenet_v3').
        pretrained (bool): Có sử dụng trọng số ImageNet hay không.
        
    Returns:
        torch.nn.Module: Mô hình đã cắt (chỉ bao gồm các lớp features).
    """
    if model_name == 'mobilenet_v3':
        try:
            from torchvision.models import MobileNet_V3_Large_Weights
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            return models.mobilenet_v3_large(weights=weights).features[:7]
        except (ImportError, AttributeError, TypeError):
            return models.mobilenet_v3_large(pretrained=pretrained).features[:7]
            
    elif model_name == 'efficientnet_b4':
        try:
            from torchvision.models import EfficientNet_B4_Weights
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            return models.efficientnet_b4(weights=weights).features[:4]
        except (ImportError, AttributeError, TypeError):
            return models.efficientnet_b4(pretrained=pretrained).features[:4]
            
    elif model_name == 'convnext_tiny':
        try:
            from torchvision.models import ConvNeXt_Tiny_Weights
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            return models.convnext_tiny(weights=weights).features[:4]
        except (ImportError, AttributeError, TypeError):
            return models.convnext_tiny(pretrained=pretrained).features[:4]
            
    else:
        raise ValueError(f"Model {model_name} không được hỗ trợ.")


def plot_result(image_raw: np.ndarray, boxes: np.ndarray, show: bool = False, save_name: Optional[str] = None, color: Tuple[int, int, int] = (255, 0, 0), label: Optional[str] = None) -> np.ndarray:
    """
    Vẽ các bounding box lên ảnh gốc và lưu file nếu được yêu cầu.
    
    Args:
        image_raw (np.ndarray): Ảnh gốc (BGR).
        boxes (np.ndarray): Mảng chứa tọa độ các bounding box.
        show (bool): Cờ quyết định có hiển thị ảnh qua matplotlib hay không.
        save_name (Optional[str]): Tên file lưu, nếu None thì không lưu.
        color (Tuple[int, int, int]): Màu của bounding box (BGR).
        label (Optional[str]): Nhãn (text) muốn vẽ lên trên bbox (vd: confidence).
        
    Returns:
        np.ndarray: Ảnh đã được vẽ bounding box.
    """
    d_img = image_raw.copy()
    for box in boxes:
        d_img = cv2.rectangle(d_img, tuple(box[0]), tuple(box[1]), color, 1)
        if label:
            x, y = box[0]
            cv2.putText(d_img, label, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if show:
        plt.imshow(d_img[:, :, ::-1])
    if save_name:
        cv2.imwrite(save_name, d_img)
    return d_img


def extract_bboxes_from_heatmap(score: np.ndarray, w_ini: int, h_ini: int, thresh: float = 0.7) -> np.ndarray:
    """
    Trích xuất các bounding box từ ma trận điểm số (heatmap) của một template và loại bỏ trùng lặp.
    
    Cách hoạt động:
    1. Lọc tương đối: Chỉ lấy những vị trí pixel có điểm số lớn hơn (thresh * điểm_cực_đại_của_template_này).
    2. Chuyển đổi các pixel ứng viên thành tọa độ bounding box.
    3. Áp dụng thuật toán giống NMS (IoU threshold = 0.5) để loại bỏ các box trùng lấn, giữ lại vị trí tốt nhất.
    
    Args:
        score (np.ndarray): Ma trận điểm số (confidence map) sinh ra từ phép convolution.
        w_ini (int): Chiều rộng của template.
        h_ini (int): Chiều cao của template.
        thresh (float): Ngưỡng tương đối (relative threshold) để lọc ứng viên ban đầu.
        
    Returns:
        np.ndarray: Mảng tọa độ các bounding box được giữ lại.
    """
    dots = np.array(np.where(score > thresh * score.max()))
    if dots.size == 0:
        return np.empty((0, 2, 2), dtype=int)

    x1 = dots[1] - w_ini // 2
    x2 = x1 + w_ini
    y1 = dots[0] - h_ini // 2
    y2 = y1 + h_ini

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = score[dots[0], dots[1]]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= 0.5)[0]
        order = order[inds + 1]

    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2, 0, 1)
    return boxes


def extract_bboxes_from_heatmap_multi(scores: np.ndarray, w_array: np.ndarray, h_array: np.ndarray, thresh_list: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trích xuất các bounding box từ ma trận điểm số của nhiều template và loại bỏ sự chồng chéo.
    
    Hàm này không thực hiện NMS tiêu chuẩn mà hoạt động qua 4 bước:
    1. Lọc template (Global relative thresholding): Bỏ qua hoàn toàn các template có điểm số cực đại nhỏ hơn 10% điểm cực đại tuyệt đối của toàn bộ tập template.
    2. Lọc ứng viên (Local relative thresholding): Trên mỗi template còn lại, lấy ra các pixel (ứng viên) có điểm số > (thresh * điểm_cực_đại_của_template_đó).
    3. Tạo Box: Chuyển các pixel ứng viên thành tọa độ bounding box và gộp tất cả lại, sắp xếp theo điểm số tuyệt đối từ cao xuống thấp.
    4. Xóa chồng chéo (Strict Suppression): Giữ lại box có điểm cao nhất, và xóa bỏ gần như hoàn toàn các box khác có diện tích giao nhau (IoU) > 0.05. Ngưỡng IoU cực thấp này giả định rằng các template (hoặc các đối tượng) không được phép đè lên nhau.
    
    Args:
        scores (np.ndarray): Mảng các ma trận điểm số (heatmaps) của tất cả template.
        w_array (np.ndarray): Chiều rộng các template tương ứng.
        h_array (np.ndarray): Chiều cao các template tương ứng.
        thresh_list (List[float]): Danh sách ngưỡng lọc cục bộ áp dụng cho từng template.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - Mảng tọa độ bounding box được giữ lại.
            - Mảng chỉ số template gốc của từng box.
            - Mảng điểm số tuyệt đối của từng box.
    """
    indices = np.arange(scores.shape[0])
    maxes = np.max(scores.reshape(scores.shape[0], -1), axis=1)
    scores_omit = scores[maxes > 0.1 * maxes.max()]
    indices_omit = indices[maxes > 0.1 * maxes.max()]

    dots = None
    dots_indices = None
    for index, score in zip(indices_omit, scores_omit):
        dot = np.array(np.where(score > thresh_list[index] * score.max()))
        if dots is None:
            dots = dot
            dots_indices = np.ones(dot.shape[-1]) * index
        else:
            dots = np.concatenate([dots, dot], axis=1)
            dots_indices = np.concatenate([dots_indices, np.ones(dot.shape[-1]) * index], axis=0)

    if dots is None or dots.size == 0:
        return np.empty((0, 2, 2), dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    dots_indices = dots_indices.astype(int)
    x1 = dots[1] - w_array[dots_indices] // 2
    x2 = x1 + w_array[dots_indices]
    y1 = dots[0] - h_array[dots_indices] // 2
    y2 = y1 + h_array[dots_indices]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = scores[dots_indices, dots[0], dots[1]]
    order = scores.argsort()[::-1]
    dots_indices = dots_indices[order]

    keep = []
    keep_index = []
    while order.size > 0:
        i = order[0]
        index = dots_indices[0]
        keep.append(i)
        keep_index.append(index)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= 0.05)[0]
        order = order[inds + 1]
        dots_indices = dots_indices[inds + 1]

    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2, 0, 1)
    confidences = scores[keep]
    return boxes, np.array(keep_index), confidences


def plot_result_multi(image_raw: np.ndarray, boxes: np.ndarray, indices: np.ndarray, show: bool = False, save_name: Optional[str] = None, color_list: Optional[List[Tuple[int, int, int]]] = None, confidences: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Vẽ kết quả của việc khớp nhiều template, tô màu riêng biệt cho từng loại template.
    
    Args:
        image_raw (np.ndarray): Ảnh gốc (BGR).
        boxes (np.ndarray): Tọa độ các bounding box.
        indices (np.ndarray): Chỉ số (index) template của từng box.
        show (bool): Cờ quyết định có hiển thị ảnh qua matplotlib hay không.
        save_name (Optional[str]): Đường dẫn lưu ảnh kết quả.
        color_list (Optional[List[Tuple[int, int, int]]]): Danh sách mã màu tùy chỉnh, nếu None sẽ tự tạo dải màu.
        confidences (Optional[np.ndarray]): Danh sách điểm số tự tin của từng box.
        
    Returns:
        np.ndarray: Ảnh đã được vẽ bounding box.
    """
    d_img = image_raw.copy()
    if len(indices) == 0:
        if show:
            plt.imshow(d_img[:, :, ::-1])
        if save_name:
            cv2.imwrite(save_name, d_img)
        return d_img

    if color_list is None:
        color_list = color_palette('hls', indices.max() + 1)
        color_list = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color_list))
    for i in range(len(indices)):
        label = f"{confidences[i]:.2f}" if confidences is not None and len(confidences) > i else None
        d_img = plot_result(d_img, boxes[i][None, :, :].copy(), color=color_list[indices[i]], label=label)
    if show:
        plt.imshow(d_img[:, :, ::-1])
    if save_name:
        cv2.imwrite(save_name, d_img)
    return d_img


def normalize_features(x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
    """
    Chuẩn hóa (Normalize) các đặc trưng để chúng nằm trên cùng một thang đo.
    """
    bs, _, H, W = x1.size()
    _, _, h, w = x2.size()
    eps = 1e-12
    x1 = x1.view(bs, -1, H * W)
    x2 = x2.view(bs, -1, h * w)
    concat = torch.cat((x1, x2), dim=2)
    x_mean = torch.mean(concat, dim=2, keepdim=True)
    x_std = torch.std(concat, dim=2, keepdim=True)
    x1 = (x1 - x_mean) / (x_std + eps)
    x2 = (x2 - x_mean) / (x_std + eps)
    x1 = x1.view(bs, -1, H, W)
    x2 = x2.view(bs, -1, h, w)
    return [x1, x2]

def compute_softmax_score(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Tính điểm số (confidence) bằng cách sử dụng softmax đa chiều.
    """
    batch_size, ref_row, ref_col, qry_row, qry_col = x.size()
    x = x.view(batch_size, ref_row * ref_col, qry_row * qry_col)
    xm_ref = x - torch.max(x, dim=1, keepdim=True)[0]
    xm_qry = x - torch.max(x, dim=2, keepdim=True)[0]
    confidence = torch.sqrt(
        F.softmax(alpha * xm_ref, dim=1) * F.softmax(alpha * xm_qry, dim=2)
    )
    conf_values, _ = torch.topk(confidence, 1)
    return conf_values.view(batch_size, ref_row, ref_col, 1)




def compute_score(x: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Tính điểm phân bổ phản hồi (response strength) thông qua bộ lọc chập.
    Giúp tránh việc nhận diện các cạnh biên ngoài của ảnh mẫu.
    
    Args:
        x (np.ndarray): Đầu vào ma trận log của confidence map.
        w (int): Chiều rộng của mẫu.
        h (int): Chiều cao của mẫu.
        
    Returns:
        np.ndarray: Ma trận điểm đã xử lý đường biên.
    """
    # Tính toán phản hồi sức mạnh cường độ (response strength)
    k = np.ones((h, w))
    score = cv2.filter2D(x, -1, k)
    score[:, :w//2] = 0
    score[:, math.ceil(-w/2):] = 0
    score[:h//2, :] = 0
    score[math.ceil(-h/2):, :] = 0
    return score



