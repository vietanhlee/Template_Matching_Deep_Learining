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

int_ = lambda x: int(round(x))

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


def nms(score: np.ndarray, w_ini: int, h_ini: int, thresh: float = 0.7) -> np.ndarray:
    """
    Thực hiện Non-maximum suppression (lọc tối đa cục bộ) cho một template.
    
    Args:
        score (np.ndarray): Ma trận điểm số (confidence map).
        w_ini (int): Chiều rộng của template.
        h_ini (int): Chiều cao của template.
        thresh (float): Ngưỡng để giữ lại bbox (so với điểm cực đại).
        
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


def nms_multi(scores: np.ndarray, w_array: np.ndarray, h_array: np.ndarray, thresh_list: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gom các ứng viên từ nhiều template rồi chạy NMS tổng thể một lần.
    
    Args:
        scores (np.ndarray): Mảng các ma trận điểm số của tất cả template.
        w_array (np.ndarray): Chiều rộng các template tương ứng.
        h_array (np.ndarray): Chiều cao các template tương ứng.
        thresh_list (List[float]): Danh sách các ngưỡng áp dụng cho từng template.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - Mảng bounding box được giữ lại.
            - Mảng chỉ số template gốc của từng box.
            - Điểm số tự tin của từng box.
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
        return np.empty((0, 2, 2), dtype=int), np.array([], dtype=int)

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


def run_one_sample(model: Any, template: torch.Tensor, image: torch.Tensor, image_name: str) -> np.ndarray:
    """
    Chấm điểm 1 template trên 1 ảnh mẫu.
    
    Args:
        model (Any): Lớp điều khiển pipeline (CreateModel).
        template (torch.Tensor): Tensor đặc trưng của ảnh template.
        image (torch.Tensor): Tensor đặc trưng của ảnh mẫu.
        image_name (str): Tên ảnh mẫu (để cache feature).
        
    Returns:
        np.ndarray: Ma trận điểm số kết quả sau khi chuẩn hóa ngược.
    """
    val = model(template, image, image_name)
    if val.is_cuda:
        val = val.cpu()
    val = val.numpy()
    val = np.log(val)

    batch_size = val.shape[0]
    scores = []
    for i in range(batch_size):
        gray = val[i, :, :, 0]
        gray = cv2.resize(gray, (image.size()[-1], image.size()[-2]))
        h = template.size()[-2]
        w = template.size()[-1]
        score = compute_score(gray, w, h)
        score[score > -1e-7] = score.min()
        score = np.exp(score / (h * w))
        scores.append(score)
    return np.array(scores)


def run_multi_sample(model: Any, dataset: ImageDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """
    Chạy toàn bộ các template trong dataset trên cùng 1 ảnh mẫu.
    
    Args:
        model (Any): Pipeline thực hiện template matching.
        dataset (ImageDataset): Dataset chứa danh sách template và 1 ảnh mẫu.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]: 
            - Ma trận điểm số tổng hợp.
            - Mảng chứa chiều rộng các template.
            - Mảng chứa chiều cao các template.
            - Danh sách các ngưỡng áp dụng cho từng template.
    """
    scores = []
    w_array = []
    h_array = []
    thresh_list = []
    for data in dataset:
        score = run_one_sample(model, data['template'], data['image'], data['image_name'])
        scores.append(score)
        w_array.append(data['template_w'])
        h_array.append(data['template_h'])
        thresh_list.append(data['thresh'])
    return np.squeeze(np.array(scores), axis=1), np.array(w_array), np.array(h_array), thresh_list


def IoU(r1: Union[List, Tuple, np.ndarray], r2: Union[List, Tuple, np.ndarray]) -> float:
    """
    Tính chỉ số Intersection over Union (IoU) giữa 2 bounding box.
    
    Args:
        r1 (Union[List, Tuple, np.ndarray]): Box 1 định dạng [x, y, w, h].
        r2 (Union[List, Tuple, np.ndarray]): Box 2 định dạng [x, y, w, h].
        
    Returns:
        float: Giá trị IoU (từ 0 đến 1).
    """
    x11, y11, w1, h1 = r1
    x21, y21, w2, h2 = r2
    x12 = x11 + w1; y12 = y11 + h1
    x22 = x21 + w2; y22 = y21 + h2
    x_overlap = max(0, min(x12,x22) - max(x11,x21) )
    y_overlap = max(0, min(y12,y22) - max(y11,y21) )
    I = 1. * x_overlap * y_overlap
    U = (y12-y11)*(x12-x11) + (y22-y21)*(x22-x21) - I
    J = I/U
    return J


def evaluate_iou(rect_gt: List, rect_pred: List) -> List[float]:
    """
    Đánh giá IoU cho một danh sách ground truth và prediction tương ứng.
    
    Args:
        rect_gt (List): Danh sách các box ground truth.
        rect_pred (List): Danh sách các box dự đoán.
        
    Returns:
        List[float]: Danh sách điểm số IoU.
    """
    # Tính điểm số iou
    score = [IoU(i, j) for i, j in zip(rect_gt, rect_pred)]
    return score


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


def locate_bbox(a: np.ndarray, w: int, h: int) -> Tuple[float, float, int, int]:
    """
    Xác định vị trí bounding box (tọa độ cực đại) dựa trên điểm phân bổ lớn nhất.
    
    Args:
        a (np.ndarray): Ma trận điểm.
        w (int): Chiều rộng template.
        h (int): Chiều cao template.
        
    Returns:
        Tuple[float, float, int, int]: Bounding box tốt nhất định dạng (x, y, w, h).
    """
    row = np.argmax(np.max(a, axis=1))
    col = np.argmax( np.max(a, axis=0) )
    x = col - 1. * w / 2
    y = row - 1. * h / 2
    return x, y, w, h


def score2curve(score: Union[List, np.ndarray], thres_delta: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chuyển đổi điểm số thành biểu đồ đánh giá (Success Rate Curve).
    
    Args:
        score (Union[List, np.ndarray]): Danh sách/Mảng điểm số.
        thres_delta (float): Bước nhảy của ngưỡng (threshold).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Ngưỡng phân bố và tỷ lệ thành công tương ứng.
    """
    thres = np.linspace(0, 1, int(1. / thres_delta) + 1)
    success_num = []
    for th in thres:
        success_num.append( np.sum(score >= (th+1e-6)) )
    success_rate = np.array(success_num) / len(score)
    return thres, success_rate


def all_sample_iou(score_list: List[np.ndarray], gt_list: List[Union[List, Tuple]]) -> List[float]:
    """
    Tính toán IoU cho tất cả các mẫu trong bộ danh sách kết quả và ground truth.
    
    Args:
        score_list (List[np.ndarray]): Danh sách ma trận điểm dự đoán.
        gt_list (List[Union[List, Tuple]]): Danh sách ground truth.
        
    Returns:
        List[float]: Danh sách mức độ chồng chéo (IoU).
    """
    num_samples = len(score_list)
    iou_list = []
    for idx in range(num_samples):
        score, image_gt = score_list[idx], gt_list[idx]
        w, h = image_gt[2:]
        pred_rect = locate_bbox( score, w, h )
        iou = IoU( image_gt, pred_rect )
        iou_list.append( iou )
    return iou_list


def plot_success_curve(iou_score: Union[List, np.ndarray], title: str = '') -> None:
    """
    Vẽ biểu đồ tỷ lệ thành công (Success Rate Curve) dựa trên điểm IoU.
    
    Args:
        iou_score (Union[List, np.ndarray]): Điểm số IoU của các dự đoán.
        title (str): Tiêu đề của biểu đồ.
    """
    thres, success_rate = score2curve(iou_score, thres_delta=0.05)
    # Tính diện tích AUC theo giao thức đánh giá truyền thống
    auc_ = np.mean(success_rate[:-1])
    plt.figure()
    plt.grid(True)
    plt.xticks(np.linspace(0,1,11))
    plt.yticks(np.linspace(0,1,11))
    plt.ylim(0, 1)
    plt.title(title + 'auc={}'.format(auc_))
    plt.plot( thres, success_rate )
    plt.show()
