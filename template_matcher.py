from pathlib import Path
import copy

import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from utils import ImageDataset

from typing import List, Tuple, Any

class Featex:
    def __init__(self, model: nn.Module, use_cuda: bool) -> None:
        """
        Khởi tạo bộ trích xuất đặc trưng (Feature Extractor) từ các mô hình backbone.
        Chỉ sử dụng 1 luồng trích xuất duy nhất (single scale) để tăng tốc độ.
        
        Args:
            model (nn.Module): Mô hình mạng CNN (MobileNetV2, EfficientNet, ConvNeXt).
            use_cuda (bool): Cờ xác định có sử dụng GPU hay không.
        """
        self.use_cuda = use_cuda
        self.model = copy.deepcopy(model.eval())
        
        for param in self.model.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.model = self.model.cuda()

    def __call__(self, input_tensor: torch.Tensor, mode: str = 'big') -> torch.Tensor:
        """
        Thực hiện trích xuất đặc trưng cho ảnh/tensor đầu vào.
        
        Args:
            input_tensor (torch.Tensor): Tensor hình ảnh đầu vào.
            mode (str): Biến dư thừa từ phiên bản cũ, giữ lại để tương thích signature.
            
        Returns:
            torch.Tensor: Tensor đặc trưng đầu ra.
        """
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
            
        return self.model(input_tensor)


class MyNormLayer:
    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        """
        Chuẩn hóa (Normalize) các đặc trưng để chúng nằm trên cùng một thang đo.
        
        Args:
            x1 (torch.Tensor): Đặc trưng 1 (ví dụ: ảnh mẫu).
            x2 (torch.Tensor): Đặc trưng 2 (ví dụ: template).
            
        Returns:
            List[torch.Tensor]: Danh sách chứa 2 tensor đã được chuẩn hóa.
        """
        # Chuẩn hóa template feature và image feature trên cùng một khoảng giá trị.
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


class CreateModel:
    def __init__(self, alpha: float, model: nn.Module, use_cuda: bool) -> None:
        """
        Khởi tạo hộp điều khiển luồng (pipeline) chính cho việc khớp mẫu.
        
        Args:
            alpha (float): Hệ số alpha điều chỉnh hàm softmax (càng lớn càng gắt).
            model (nn.Module): Backbone CNN (VGG19).
            use_cuda (bool): Sử dụng GPU.
        """
        # Đây là bộ điều khiển luồng (pipeline): feature extraction -> matching -> confidence.
        self.alpha = alpha
        self.featex = Featex(model, use_cuda)
        self.I_feat = None
        self.I_feat_name = None

    def __call__(self, template: torch.Tensor, image: torch.Tensor, image_name: str) -> torch.Tensor:
        """
        Tính toán ma trận độ tin cậy (confidence map) giữa template và ảnh mẫu.
        
        Args:
            template (torch.Tensor): Tensor ảnh template.
            image (torch.Tensor): Tensor ảnh mẫu lớn.
            image_name (str): Tên file của ảnh mẫu để tận dụng bộ nhớ đệm (cache).
            
        Returns:
            torch.Tensor: Ma trận điểm tin cậy (confidence maps).
        """
        # Template được xử lý lại mỗi lần, còn ảnh (image) chỉ tính toán lại đặc trưng khi tên ảnh thay đổi.
        T_feat = self.featex(template)
        if self.I_feat_name != image_name:
            self.I_feat = self.featex(image)
            self.I_feat_name = image_name

        conf_maps = None
        batchsize_T = T_feat.size()[0]
        for i in range(batchsize_T):
            T_feat_i = T_feat[i].unsqueeze(0)
            I_feat_norm, T_feat_i = MyNormLayer()(self.I_feat, T_feat_i)
            # Hàm einsum tạo ma trận tương đồng (similarity) giữa từng điểm của ảnh sample và template.
            dist = torch.einsum(
                'xcab,xcde->xabde',
                I_feat_norm / torch.norm(I_feat_norm, dim=1, keepdim=True),
                T_feat_i / torch.norm(T_feat_i, dim=1, keepdim=True),
            )
            conf_map = MatchingScorer(self.alpha)(dist)
            if conf_maps is None:
                conf_maps = conf_map
            else:
                conf_maps = torch.cat([conf_maps, conf_map], dim=0)
        return conf_maps


class MatchingScorer:
    def __init__(self, alpha: float) -> None:
        """
        Khởi tạo lớp tính điểm tương đồng.
        
        Args:
            alpha (float): Hệ số nhạy của Softmax.
        """
        self.alpha = alpha

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tính điểm số (confidence) bằng cách sử dụng softmax đa chiều.
        
        Args:
            x (torch.Tensor): Tensor đầu vào biểu diễn độ tương quan cosine.
            
        Returns:
            torch.Tensor: Tensor điểm số cuối cùng.
        """
        # Chuyển đổi sang mảng 2 chiều, sau đó tính confidence theo cả phía template và phía query.
        batch_size, ref_row, ref_col, qry_row, qry_col = x.size()
        x = x.view(batch_size, ref_row * ref_col, qry_row * qry_col)
        xm_ref = x - torch.max(x, dim=1, keepdim=True)[0]
        xm_qry = x - torch.max(x, dim=2, keepdim=True)[0]
        confidence = torch.sqrt(
            F.softmax(self.alpha * xm_ref, dim=1) * F.softmax(self.alpha * xm_qry, dim=2)
        )
        conf_values, _ = torch.topk(confidence, 1)
        return conf_values.view(batch_size, ref_row, ref_col, 1)

    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Hỗ trợ tính toán kích thước đầu ra mong muốn.
        
        Args:
            input_shape (Tuple): Kích thước đầu vào.
            
        Returns:
            Tuple: Kích thước đầu ra theo format (batch_size, H, W, 1).
        """
        bs, H, W, _, _ = input_shape
        return (bs, H, W, 1)
