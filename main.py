import argparse
import os
from typing import List

import cv2

from models import ModelManager
from template_matcher import MATCH_METHODS, TemplateMatcher
from utils import bgr_to_rgb, color_from_name, draw_results, list_images, load_image_bgr, to_json


def parse_args() -> argparse.Namespace:
    """Phân tích tham số dòng lệnh cho CLI.

    Returns:
        Namespace chứa các tham số đã parse.
    """
    parser = argparse.ArgumentParser(description="Template matching CLI")
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh đầu vào")
    parser.add_argument("--templates", nargs="*", default=[], help="Danh sách đường dẫn template")
    parser.add_argument("--template-dir", default="template", help="Thư mục chứa template")
    parser.add_argument("--output-image", default="output.png", help="Đường dẫn ảnh kết quả")
    parser.add_argument("--output-json", default="output.json", help="Đường dẫn JSON kết quả")
    parser.add_argument("--match-threshold", type=float, default=0.7)
    parser.add_argument("--cosine-threshold", type=float, default=0.75)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument(
        "--match-method",
        default="TM_CCOEFF_NORMED",
        choices=sorted(MATCH_METHODS.keys()),
        help="Phương pháp matchTemplate",
    )
    parser.add_argument(
        "--angles",
        default="0,45,90,135,180,225,270,315",
        help="Danh sách góc xoay, cách nhau bởi dấu phẩy",
    )
    parser.add_argument("--scale-min", type=float, default=0.1)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--scale-steps", type=int, default=10)
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--max-detections", type=int, default=50, help="Giới hạn bbox mỗi template")
    parser.add_argument("--no-mt", action="store_true", help="Tắt đa luồng")
    parser.add_argument("--no-cnn", action="store_true", help="Tắt xác nhận bằng CNN")
    return parser.parse_args()


def collect_templates(paths: List[str], template_dir: str) -> List[str]:
    """Gộp danh sách template từ đường dẫn chỉ định và thư mục.

    Args:
        paths: Danh sách đường dẫn template được truyền trực tiếp.
        template_dir: Thư mục chứa template bổ sung.

    Returns:
        Danh sách đường dẫn duy nhất (đã loại trùng).
    """
    templates = list(paths)
    if template_dir and os.path.isdir(template_dir):
        # Nếu cung cấp `template_dir`, thêm các file trong thư mục vào danh sách.
        # Thao tác tách riêng để cho phép kết hợp template truyền tay và trong thư mục.
        templates.extend(list_images(template_dir))
    unique = []
    seen = set()
    for p in templates:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def main() -> None:
    """Chạy pipeline template matching qua CLI.

    Raises:
        SystemExit: Khi không có template nào được cung cấp.
    """
    args = parse_args()

    image_bgr = load_image_bgr(args.image)
    template_paths = collect_templates(args.templates, args.template_dir)
    if not template_paths:
        raise SystemExit("Không có template nào được cung cấp")

    templates = {os.path.basename(p): load_image_bgr(p) for p in template_paths}

    angles = [float(x.strip()) for x in args.angles.split(",") if x.strip()]
    matcher = TemplateMatcher(
        match_threshold=args.match_threshold,
        iou_threshold=args.iou_threshold,
        match_method=args.match_method,
        angles=angles,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        scale_steps=args.scale_steps,
        max_detections_per_template=args.max_detections,
        use_multithreading=not args.no_mt,
    )

    results = matcher.find(image_bgr, templates)

    if not args.no_cnn:
        model_manager = ModelManager()
        filtered = []
        for r in results:
            x, y, w, h = r.bbox
            crop = image_bgr[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            template_resized = cv2.resize(r.template_variant, (w, h))
            crop_rgb = bgr_to_rgb(crop)
            template_rgb = bgr_to_rgb(template_resized)
            emb_a = model_manager.embed(args.model, crop_rgb)
            emb_b = model_manager.embed(args.model, template_rgb)
            sim = model_manager.cosine_similarity(emb_a, emb_b)
            r.cosine_similarity = sim
            if sim >= args.cosine_threshold:
                filtered.append(r)
        results = filtered

    json_items = [r.to_dict() for r in results]
    color_map = {name: color_from_name(name) for name in templates.keys()}
    output_image = draw_results(image_bgr, json_items, color_map)

    cv2.imwrite(args.output_image, output_image)
    with open(args.output_json, "w", encoding="utf-8") as f:
        f.write(to_json(json_items))


if __name__ == "__main__":
    main()
