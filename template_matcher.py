import copy
from typing import List, Tuple, Any, Dict, Union
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils import (
    ImageDataset, 
    build_feature_extractor, 
    extract_bboxes_from_heatmap_multi, 
    plot_result_multi, 
    normalize_features, 
    compute_softmax_score, 
    compute_score
)

class TemplateMatcher:
    def __init__(self, model: nn.Module, use_cuda: bool = True) -> None:
        """
        Khởi tạo hệ thống khớp mẫu (Template Matcher).
        
        Args:
            model (nn.Module): Backbone CNN đã được khởi tạo.
            use_cuda (bool): Sử dụng GPU.
        """
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.model = copy.deepcopy(model.eval())
        
        for param in self.model.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.model = self.model.cuda()
            
        self.I_feat = None
        self.I_feat_name = None

    def _extract_features(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        return self.model(input_tensor)

    def compute_confidence_map(self, template: torch.Tensor, image: torch.Tensor, image_name: str, alpha: float) -> torch.Tensor:
        T_feat = self._extract_features(template)
        if self.I_feat_name != image_name:
            self.I_feat = self._extract_features(image)
            self.I_feat_name = image_name

        conf_maps = None
        batchsize_T = T_feat.size()[0]
        for i in range(batchsize_T):
            T_feat_i = T_feat[i].unsqueeze(0)
            I_feat_norm, T_feat_i = normalize_features(self.I_feat, T_feat_i)
            dist = torch.einsum(
                'xcab,xcde->xabde',
                I_feat_norm / torch.norm(I_feat_norm, dim=1, keepdim=True),
                T_feat_i / torch.norm(T_feat_i, dim=1, keepdim=True),
            )
            conf_map = compute_softmax_score(dist, alpha)
            if conf_maps is None:
                conf_maps = conf_map
            else:
                conf_maps = torch.cat([conf_maps, conf_map], dim=0)
        return conf_maps

    def _run_one_sample(self, template: torch.Tensor, image: torch.Tensor, image_name: str, alpha: float) -> np.ndarray:
        val = self.compute_confidence_map(template, image, image_name, alpha)
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

    def find(self, sample_image_path: str, templates_dir: str, alpha: float = 20, thresh: float = 0.2, conf_thresh: float = 0.07, template_scale: float = 1.0) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Thực hiện so khớp mẫu và trả về kết quả ảnh kèm JSON.
        
        Args:
            sample_image_path (str): Đường dẫn ảnh gốc.
            templates_dir (str): Đường dẫn thư mục chứa các ảnh template.
            alpha (float): Hệ số alpha điều chỉnh hàm softmax (temperature scaling).
            thresh (float): Ngưỡng NMS tương đối.
            conf_thresh (float): Ngưỡng loại bỏ tuyệt đối.
            template_scale (float): Tỉ lệ thu phóng ảnh template.
            
        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: Ảnh kết quả (BGR) và danh sách detections định dạng JSON.
        """
        dataset = ImageDataset(templates_dir, sample_image_path, thresh=thresh, template_scale=template_scale)
        
        scores = []
        w_array = []
        h_array = []
        thresh_list = []
        for data in dataset:
            score = self._run_one_sample(data['template'], data['image'], data['image_name'], alpha)
            scores.append(score)
            w_array.append(data['template_w'])
            h_array.append(data['template_h'])
            thresh_list.append(data['thresh'])
            
        scores_arr = np.squeeze(np.array(scores), axis=1)
        w_array_arr = np.array(w_array)
        h_array_arr = np.array(h_array)
        
        boxes, indices, confidences = extract_bboxes_from_heatmap_multi(scores_arr, w_array_arr, h_array_arr, thresh_list)

        # Lọc các bbox theo ngưỡng confidence tuyệt đối
        filtered_boxes = []
        filtered_indices = []
        filtered_confidences = []
        for i in range(len(boxes)):
            conf = float(confidences[i]) if len(confidences) > i else 0.0
            if conf >= conf_thresh:
                filtered_boxes.append(boxes[i])
                filtered_indices.append(indices[i])
                filtered_confidences.append(confidences[i])
                
        boxes = np.array(filtered_boxes) if len(filtered_boxes) > 0 else np.empty((0, 2, 2))
        indices = np.array(filtered_indices, dtype=int)
        confidences = np.array(filtered_confidences)

        # Chuẩn bị dữ liệu JSON
        detections = []
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1 = int(box[0][0]), int(box[0][1])
            x2, y2 = int(box[1][0]), int(box[1][1])
            tpl_idx_raw = int(indices[i]) if len(indices) > i else None
            tpl_idx = tpl_idx_raw // len(dataset.angles) if tpl_idx_raw is not None else None
            angle = dataset.angles[tpl_idx_raw % len(dataset.angles)] if tpl_idx_raw is not None else None
            try:
                tpl_name = f"{Path(dataset.template_paths[tpl_idx_raw]).name} (Angle: {angle}°)"
            except Exception:
                tpl_name = None
            conf = float(confidences[i]) if len(confidences) > i else None
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'template_index': tpl_idx,
                'angle': angle,
                'template_name': tpl_name,
                'confidence': conf,
            })

        base_indices = indices // len(dataset.angles) if len(indices) > 0 else np.array([], dtype=int)
        result_bgr = plot_result_multi(dataset.image_raw, boxes, base_indices, show=False, confidences=confidences)
        
        return result_bgr, detections
