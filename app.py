import os
from typing import Dict, List

import cv2
import gradio as gr
import numpy as np

from models import ModelManager
from template_matcher import MatchResult, TemplateMatcher
from utils import (
    bgr_to_rgb,
    color_from_name,
    draw_results,
    is_mostly_white,
    list_images,
    load_image_bgr,
    rgb_to_bgr,
)


SAMPLE_DIR = "sample"
TEMPLATE_DIR = "template"
MODEL_CHOICES = ["convnext_tiny", "efficientnet_b4", "mobilenetv3_large_100"]
MATCH_METHOD_CHOICES = ["TM_CCOEFF_NORMED", "TM_CCORR_NORMED"]
ANGLES_DEFAULT = [0, 45, 90, 135, 180, 225, 270, 315]
DEFAULT_MODEL = "efficientnet_b4"


model_manager = ModelManager()
# Preload model mặc định để tránh độ trễ tải model khi lần chạy đầu tiên
# (việc tải model timm có thể mất thời gian, nên gọi sẵn khi khởi tạo app).
model_manager.get(DEFAULT_MODEL)


def _filelist_to_paths(files) -> List[str]:
    """Chuyển danh sách file upload từ Gradio sang danh sách đường dẫn.

    Args:
        files: Danh sách file Gradio (chuỗi đường dẫn hoặc đối tượng có thuộc tính `name`).

    Returns:
        Danh sách đường dẫn hợp lệ.
    """
    if not files:
        return []
    paths = []
    for f in files:
        if isinstance(f, str):
            paths.append(f)
        elif hasattr(f, "name"):
            paths.append(f.name)
    return paths


def _load_templates(selected_paths: List[str], uploads) -> Dict[str, np.ndarray]:
    """Nạp template từ danh sách đã chọn và file upload.

    Args:
        selected_paths: Danh sách đường dẫn template đã chọn.
        uploads: Danh sách file upload từ Gradio.

    Returns:
        Từ điển {tên_template: ảnh_bgr}.
    """
    paths = list(selected_paths or [])
    paths.extend(_filelist_to_paths(uploads))
    templates = {}
    for path in paths:
        if not path:
            continue
        name = os.path.basename(path)
        templates[name] = load_image_bgr(path)
    return templates


def run_matching(
    image_rgb,
    template_selected_paths,
    template_uploads,
    match_threshold: float,
    cosine_threshold: float,
    iou_threshold: float,
    match_method: str,
    scale_min: float,
    scale_max: float,
    scale_steps: int,
    max_detections: int,
    use_multithreading: bool,
    use_cnn: bool,
    model_name: str,
):
    """Chạy template matching và (tuy chon) xac nhan bang CNN.

    Args:
        image_rgb: Ảnh đầu vào dạng RGB (numpy array).
        template_selected_paths: Đường dẫn template được chọn từ gallery.
        template_uploads: Danh sách file template upload.
        match_threshold: Ngưỡng điểm `matchTemplate`.
        cosine_threshold: Ngưỡng cosine khi xác nhận bằng CNN.
        iou_threshold: Ngưỡng IoU cho NMS.
        match_method: Tên phương pháp `matchTemplate`.
        scale_min: Tỉ lệ scale nhỏ nhất.
        scale_max: Tỉ lệ scale lớn nhất.
        scale_steps: Số bước scale.
        max_detections: Giới hạn số bbox mỗi template.
        use_multithreading: Bật/tắt đa luồng khi match.
        use_cnn: Bật/tắt xác nhận bằng CNN.
        model_name: Tên model timm để lấy embedding.

    Returns:
        2-tuple: (ảnh kết quả dạng RGB, danh sách kết quả dạng dict).

    Raises:
        gr.Error: Khi thiếu ảnh đầu vào hoặc template.
    """
    if image_rgb is None:
        raise gr.Error("Vui lòng chọn hoặc upload ảnh cần so sánh")

    image_bgr = rgb_to_bgr(image_rgb)
    templates = _load_templates(template_selected_paths, template_uploads)
    if not templates:
        raise gr.Error("Vui lòng chọn hoặc upload ảnh template")

    matcher = TemplateMatcher(
        match_threshold=match_threshold,
        iou_threshold=iou_threshold,
        match_method=match_method,
        angles=ANGLES_DEFAULT,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_steps=scale_steps,
        max_detections_per_template=max_detections,
        use_multithreading=use_multithreading,
    )

    results: List[MatchResult] = matcher.find(image_bgr, templates)
    if results:
        filtered_blank: List[MatchResult] = []
        for r in results:
            x, y, w, h = r.bbox
            crop = image_bgr[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            # Lọc bỏ các crop gần như trắng (không có nội dung hữu ích)
            # để tránh false positive do nền trắng hoặc vùng rỗng.
            if is_mostly_white(crop):
                continue
            filtered_blank.append(r)
        results = filtered_blank

    if use_cnn and results:
        model, transform = model_manager.get(model_name)
        filtered: List[MatchResult] = []
        for r in results:
            x, y, w, h = r.bbox
            crop = image_bgr[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            # Trước khi xác nhận bằng CNN, loại bỏ các vùng trắng và
            # chuẩn hóa kích thước template_variant về đúng kích thước bbox.
            if is_mostly_white(crop):
                continue
            template_resized = cv2.resize(r.template_variant, (w, h))
            crop_rgb = bgr_to_rgb(crop)
            template_rgb = bgr_to_rgb(template_resized)
            emb_a = model_manager.embed_with(model, transform, crop_rgb)
            emb_b = model_manager.embed_with(model, transform, template_rgb)
            sim = model_manager.cosine_similarity(emb_a, emb_b)
            r.cosine_similarity = sim
            if sim >= cosine_threshold:
                filtered.append(r)
        results = filtered

    json_items = [r.to_dict() for r in results]
    color_map = {name: color_from_name(name) for name in templates.keys()}
    output_bgr = draw_results(image_bgr, json_items, color_map)
    output_rgb = bgr_to_rgb(output_bgr)
    return output_rgb, json_items


sample_paths = list_images(SAMPLE_DIR)
template_files = list_images(TEMPLATE_DIR)
template_choices = [(os.path.basename(p), p) for p in template_files]


def _toggle_template_selection(evt: gr.SelectData, selected: List[str]) -> List[str]:
    """Bật/tắt lựa chọn template theo thao tác click trong gallery.

    Args:
        evt: Sự kiện chọn từ gallery (gr.SelectData).
        selected: Danh sách đường dẫn đang được chọn.

    Returns:
        Danh sách đường dẫn sau khi cập nhật.
    """
    if evt is None or evt.index is None:
        return selected
    idx = int(evt.index)
    if idx < 0 or idx >= len(template_files):
        return selected
    path = template_files[idx]
    if path in selected:
        selected = [p for p in selected if p != path]
    else:
        selected = selected + [path]
    return selected


def _clear_template_selection() -> List[str]:
    """Xóa toàn bộ lựa chọn template hiện tại.

    Returns:
        Danh sách rỗng.
    """
    return []

CUSTOM_CSS = """
:root {
    --bg: #0f172a;
    --panel: #111c36;
    --panel-soft: #152243;
    --accent: #ffb703;
    --accent-2: #06d6a0;
    --text: #e6edf7;
    --muted: #9fb0c7;
    --stroke: #22345a;
    --shadow: 0 10px 30px rgba(6, 10, 30, 0.35);
    --radius: 16px;
    --radius-sm: 12px;
    --mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
    --display: "Bricolage Grotesque", "Space Grotesk", "Segoe UI", sans-serif;
}

.gradio-container {
    background: radial-gradient(1200px 600px at 10% -10%, #1b2b57 0%, transparent 60%),
                            radial-gradient(900px 500px at 100% 0%, #3a1d52 0%, transparent 55%),
                            var(--bg);
    color: var(--text);
    font-family: var(--display);
}

.app-shell {
    max-width: 1200px;
    margin: 0 auto;
    padding: 24px 20px 40px;
}

.hero {
    background: linear-gradient(130deg, rgba(255, 183, 3, 0.18), rgba(6, 214, 160, 0.1));
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: var(--shadow);
    border-radius: var(--radius);
    padding: 20px 24px;
    margin-bottom: 18px;
}

.hero h1 {
    font-size: 30px;
    margin: 0 0 6px 0;
    letter-spacing: 0.2px;
}

.hero p {
    margin: 0;
    color: var(--muted);
}

.section-title {
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 12px;
    color: var(--muted);
    margin: 18px 0 10px 6px;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--stroke);
    border-radius: var(--radius);
    padding: 16px;
    box-shadow: var(--shadow);
}

.panel-soft {
    background: var(--panel-soft);
    border: 1px solid var(--stroke);
    border-radius: var(--radius-sm);
    padding: 14px;
}

.config-note {
    font-family: var(--mono);
    font-size: 12px;
    line-height: 1.45;
    color: var(--muted);
    background: rgba(10, 18, 40, 0.6);
    border: 1px dashed rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-sm);
    padding: 12px 14px;
}

.template-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin: 8px 0 10px;
}

.template-badge {
    font-size: 12px;
    color: var(--muted);
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 6px 10px;
    border-radius: 999px;
}

.gr-button-primary {
    background: linear-gradient(135deg, #ffb703, #f48c06);
    border: none;
    color: #1a1202;
    font-weight: 700;
}

.gr-button-primary:hover {
    filter: brightness(1.05);
}

.template-gallery .thumbnail {
    border-radius: 12px;
    border: 2px solid transparent;
    overflow: hidden;
}

.template-gallery .thumbnail:hover {
    border-color: rgba(255, 183, 3, 0.65);
}

.footer-hint {
    color: var(--muted);
    font-size: 12px;
    margin-top: 8px;
}
"""

with gr.Blocks(title="Template Matching") as demo:
    with gr.Column(elem_classes="app-shell"):
        gr.Markdown(
            """
<div class="hero">
    <h1>Khớp mẫu ảnh thông minh</h1>
    <p>So khớp đa tỉ lệ, đa góc xoay và tùy chọn xác nhận bằng CNN để giảm false positive.</p>
</div>
"""
        )

        gr.Markdown("<div class='section-title'>Cấu hình nhanh</div>")
        gr.Markdown(
            """
<div class="config-note">
Ngưỡng template matching cao hơn giúp giảm false positive nhưng có thể bỏ sót đối tượng nhỏ.
IoU NMS thấp giúp giảm trùng lặp mạnh hơn. Tăng số bước scale để chính xác hơn nhưng chậm hơn.
CNN giúp lọc nhiễu nhưng cần thời gian suy luận.
</div>
"""
        )

        gr.Markdown("<div class='section-title'>Dữ liệu vào</div>")
        with gr.Row():
            with gr.Column(scale=5, elem_classes="panel"):
                image_input = gr.Image(
                    label="Ảnh cần so sánh",
                    type="numpy",
                )
                gr.Examples(sample_paths, inputs=image_input, label="Ảnh mẫu (sample)")

            with gr.Column(scale=7, elem_classes="panel"):
                gr.Markdown("Chọn template bằng cách bấm vào ảnh trong gallery.")
                template_uploads = gr.Files(label="Upload template (nhiều file)")
                with gr.Row(elem_classes="template-toolbar"):
                    template_selected = gr.Dropdown(
                        choices=template_choices,
                        value=[],
                        multiselect=True,
                        label="Template đã chọn",
                    )
                    clear_templates_btn = gr.Button("Xóa lựa chọn")
                if template_files:
                    template_gallery = gr.Gallery(
                        value=[(p, os.path.basename(p)) for p in template_files],
                        label="Template trong thư mục template",
                        columns=4,
                        height=260,
                        elem_classes="template-gallery",
                    )
                    template_gallery.select(
                        _toggle_template_selection,
                        inputs=template_selected,
                        outputs=template_selected,
                    )
                clear_templates_btn.click(
                    _clear_template_selection,
                    inputs=None,
                    outputs=template_selected,
                )

        gr.Markdown("<div class='section-title'>Cấu hình OpenCV</div>")
        with gr.Row():
            match_threshold = gr.Slider(
                0.1,
                1.0,
                value=0.73,
                step=0.01,
                label="Ngưỡng template matching",
            )
            match_method = gr.Dropdown(
                choices=MATCH_METHOD_CHOICES,
                value="TM_CCOEFF_NORMED",
                label="Phương pháp matchTemplate",
            )
            iou_threshold = gr.Slider(
                0.1,
                0.9,
                value=0.15,
                step=0.05,
                label="IoU NMS",
            )

        with gr.Row():
            scale_min = gr.Slider(
                0.1,
                1.0,
                value=0.2,
                step=0.05,
                label="Scale min",
            )
            scale_max = gr.Slider(
                1.0,
                4.0,
                value=2,
                step=0.1,
                label="Scale max",
            )
            scale_steps = gr.Slider(
                2,
                50,
                value=50,
                step=1,
                label="Số bước scale",
            )

        with gr.Row():
            max_detections = gr.Slider(
                1,
                200,
                value=100,
                step=1,
                label="Giới hạn bbox mỗi template",
            )
            use_multithreading = gr.Checkbox(
                value=True,
                label="Dùng đa luồng",
            )

        gr.Markdown("<div class='section-title'>Xác nhận bằng CNN</div>")
        with gr.Row():
            cosine_threshold = gr.Slider(
                0.1,
                1.0,
                value=0.78,
                step=0.01,
                label="Ngưỡng cosine",
            )
            use_cnn = gr.Checkbox(
                value=True,
                label="Sử dụng CNN để xác nhận",
            )
            model_name = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=DEFAULT_MODEL,
                label="Mô hình CNN",
            )

        run_btn = gr.Button("Chạy matching", variant="primary")

        gr.Markdown("<div class='section-title'>Kết quả</div>")
        with gr.Row():
            output_image = gr.Image(label="Kết quả", type="numpy")
            output_json = gr.JSON(label="JSON kết quả")

        gr.Markdown("<div class='footer-hint'>Tip: tăng ngưỡng cosine nếu muốn lọc gắt hơn.</div>")

    run_btn.click(
        fn=run_matching,
        inputs=[
            image_input,
            template_selected,
            template_uploads,
            match_threshold,
            cosine_threshold,
            iou_threshold,
            match_method,
            scale_min,
            scale_max,
            scale_steps,
            max_detections,
            use_multithreading,
            use_cnn,
            model_name,
        ],
        outputs=[output_image, output_json],
    )


if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)
