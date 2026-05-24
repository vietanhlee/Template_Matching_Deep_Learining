from __future__ import annotations

import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

from typing import Any, List, Tuple, Dict, Union

from template_matcher import TemplateMatcher
from utils import build_feature_extractor


def _resolve_file_path(file_item: Any) -> str:
    """
    Chuyển đổi đối tượng file từ Gradio về đường dẫn file thực tế trên ổ cứng.
    
    Args:
        file_item (Any): Đối tượng file nhận được từ Gradio (có thể là chuỗi, PathLike, dict, v.v.).
        
    Returns:
        str: Đường dẫn tuyệt đối đến file.
    """
    if file_item is None:
        raise ValueError('Template file is missing.')
    if isinstance(file_item, str):
        return file_item
    if isinstance(file_item, os.PathLike):
        return os.fspath(file_item)
    if isinstance(file_item, dict):
        for key in ('path', 'name', 'orig_name'):
            value = file_item.get(key)
            if value:
                return value
    if hasattr(file_item, 'name'):
        return file_item.name
    raise TypeError(f'Unsupported file type: {type(file_item)!r}')


@lru_cache(maxsize=4)
def _build_backbone(model_name: str) -> torch.nn.Module:
    """
    Khởi tạo và bộ nhớ đệm (cache) mô hình backbone để trích xuất đặc trưng.
    
    Args:
        model_name (str): Tên mô hình (ví dụ: 'convnext_tiny').
        
    Returns:
        torch.nn.Module: Mô hình đã được pretrained.
    """
    return build_feature_extractor(model_name, pretrained=True)


def _run_matching(
    sample_image: np.ndarray, 
    template_files: Union[List[Any], Any], 
    alpha: float, 
    thresh: float,
    conf_thresh: float,
    template_scale: float,
    model_name: str
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Chạy thuật toán Template Matching trên một ảnh mẫu và nhiều ảnh template.
    
    Args:
        sample_image (np.ndarray): Ảnh mẫu đầu vào dưới dạng mảng numpy (RGB).
        template_files (Union[List[Any], Any]): Danh sách các file template tải lên.
        alpha (float): Hệ số alpha điều chỉnh độ "gắt" của hàm softmax.
        thresh (float): Ngưỡng độ tin cậy để lọc các bounding box.
        model_name (str): Tên backbone model.
        
    Returns:
        Tuple[np.ndarray, List[Dict[str, Any]]]: 
            - Ảnh kết quả đã vẽ bounding box (RGB).
            - Danh sách các dự đoán (chứa tọa độ bbox, tên template và độ tự tin).
    """
    import numpy as np  # Ensure numpy is imported for type hints if not already
    # Gradio tự động ném ra Exception này khi code thực thi raise gr.Error. 
    # Bắt lỗi này là hoàn toàn cố ý và chuẩn mực để hiển thị thông báo popup màu đỏ trên giao diện thay vì crash.
    if sample_image is None:
        raise gr.Error('Vui lòng tải lên 1 ảnh mẫu (sample) để bắt đầu.')
    if not template_files:
        raise gr.Error('Vui lòng tải lên ít nhất 1 ảnh template.')

    if isinstance(template_files, dict):
        template_files = [template_files]

    use_cuda = torch.cuda.is_available()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        sample_bgr = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)
        sample_path = temp_dir_path / 'sample.png'
        cv2.imwrite(str(sample_path), sample_bgr)

        # Sao chép các ảnh template vào một thư mục con riêng biệt để ImageDataset
        # không bao gồm ảnh sample khi lặp qua các file template.
        templates_dir = temp_dir_path / 'templates'
        templates_dir.mkdir()
        for index, file_item in enumerate(template_files):
            source_path = Path(_resolve_file_path(file_item))
            destination_path = templates_dir / f'template_{index}{source_path.suffix or ".png"}'
            shutil.copy2(source_path, destination_path)

        matcher = TemplateMatcher(model=_build_backbone(model_name), use_cuda=use_cuda)
        result_bgr, detections = matcher.find(
            sample_image_path=str(sample_path),
            templates_dir=str(templates_dir),
            alpha=alpha,
            thresh=thresh,
            conf_thresh=conf_thresh,
            template_scale=template_scale
        )

        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        return result_rgb, detections


def build_app() -> gr.Blocks:
    """
    Xây dựng giao diện Gradio cho ứng dụng CNN Template Matching.
    
    Returns:
        gr.Blocks: Khối giao diện Gradio hoàn chỉnh.
    """
    with gr.Blocks(title='CNN Template Matching') as demo:
        gr.Markdown('# CNN Template Matching')
        gr.Markdown(
            'Tải lên ảnh bản vẽ và ảnh các template cần tìm trong bản vẽ đó. '
            'Bạn có thể tùy chọn mô hình trích xuất đặc trưng (Backbone) để cân bằng giữa tốc độ và độ chính xác (mặc định dùng ConvNeXt-Tiny).'
        )

        with gr.Row():
            with gr.Column():
                sample_image = gr.Image(label='Sample image', type='numpy')
                gr.Examples(
                    examples=[["sample/sample2.png"]],
                    inputs=sample_image,
                    label="Ảnh mẫu (Sample) ví dụ"
                )
            
            with gr.Column():
                template_files = gr.File(
                    label='Templates',
                    file_count='multiple',
                    file_types=['image'],
                )
                
                dummy_image = gr.Image(type="filepath", visible=False)
                
                gr.Examples(
                    examples=[
                        ["template/template_2_1.png"],
                        ["template/template_2_2.png"],
                        ["template/template_2_4.png"],
                        ["template/template_2_3.png"]
                    ],
                    inputs=dummy_image,
                    label="Các Template ví dụ (Nhấp vào ảnh để thêm vào danh sách)"
                )
                
                def _add_example_template(new_img_path: str, current_files: Any) -> List[str]:
                    if not new_img_path:
                        return current_files if current_files is not None else []
                    
                    paths = []
                    if current_files:
                        if not isinstance(current_files, list):
                            current_files = [current_files]
                        for f in current_files:
                            try:
                                paths.append(_resolve_file_path(f))
                            except Exception:
                                pass
                                
                    if new_img_path not in paths:
                        paths.append(new_img_path)
                    return paths

                dummy_image.change(
                    fn=_add_example_template,
                    inputs=[dummy_image, template_files],
                    outputs=[template_files]
                )

        with gr.Row():
            alpha = gr.Slider(minimum=1, maximum=100, value= 20, step= 0.5, label='Alpha', info='Hệ số điều chỉnh softmax (alpha càng lớn, kết quả matching càng "gắt" và loại bỏ noise tốt hơn nhưng dễ hụt object).')
            thresh = gr.Slider(minimum=0.1, maximum=1.0, value=0.2, step=0.01, label='Local Relative Threshold', info='Ngưỡng lọc đỉnh cục bộ (nhân với điểm max của template) để tìm ứng viên (Mặc định: 0.2).')
            conf_thresh = gr.Slider(minimum=0.0, maximum=1.0, value=0.065, step=0.005, label='Confidence Threshold', info='Ngưỡng độ tin cậy tuyệt đối để lọc bbox sau cùng (Mặc định: 0.5).')
            template_scale = gr.Slider(minimum=0.1, maximum=3.0, value=1.0, step=0.1, label='Template Scale', info='Tỉ lệ thu phóng template. Giảm < 1.0 nếu template to hơn thực tế, tăng > 1.0 nếu nhỏ hơn.')
            model_name = gr.Dropdown(choices=['convnext_tiny', 'efficientnet_b4', 'mobilenet_v3'], value='efficientnet_b4', label='Backbone Model', info='Chọn mô hình trích xuất đặc trưng.')

        run_button = gr.Button('Run Template Matching')
        output_image = gr.Image(label='BBox result')
        json_output = gr.JSON(label='Detections')

        run_button.click(
            fn=_run_matching,
            inputs=[sample_image, template_files, alpha, thresh, conf_thresh, template_scale, model_name],
            outputs=[output_image, json_output],
        )

    return demo


if __name__ == '__main__':
    build_app().launch(ssr_mode=False)