import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


def list_images(folder: str, extensions: Optional[Sequence[str]] = None) -> List[str]:
    """Liệt kê các ảnh trong thư mục theo phần mở rộng cho trước."""
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    paths: List[str] = []
    if not os.path.isdir(folder):
        return paths
    for name in os.listdir(folder):
        _, ext = os.path.splitext(name)
        if ext.lower() in extensions:
            paths.append(os.path.join(folder, name))
    return sorted(paths)


def load_image_bgr(path: str) -> np.ndarray:
    """Đọc ảnh bằng OpenCV và trả về ảnh dạng BGR."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return image


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Chuyển ảnh từ BGR sang RGB."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    """Chuyển ảnh từ RGB sang BGR."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Xoay ảnh theo góc bất kỳ và giữ nguyên toàn bộ nội dung."""
    if angle % 360 == 0:
        return image
    (h, w) = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR)


def trim_white_border(
    image_bgr: np.ndarray,
    threshold: int = 245,
    min_size: int = 5,
) -> np.ndarray:
    """Cắt viền trắng dư thừa của template trước khi xử lý."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray < threshold
    if not np.any(mask):
        return image_bgr
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cropped = image_bgr[y0 : y1 + 1, x0 : x1 + 1]
    if cropped.shape[0] < min_size or cropped.shape[1] < min_size:
        return image_bgr
    return cropped


def compute_scale_candidates(
    image_shape: Tuple[int, int, int],
    template_shape: Tuple[int, int, int],
    min_scale: float,
    max_scale: float,
    steps: int,
) -> List[float]:
    """Tính danh sách tỉ lệ scale sao cho template vẫn nằm trong ảnh."""
    image_h, image_w = image_shape[:2]
    template_h, template_w = template_shape[:2]
    max_allowed = min(image_w / template_w, image_h / template_h)
    if max_allowed <= 0:
        return []
    scale_max = min(max_scale, max_allowed)
    scale_min = min_scale
    if scale_max < scale_min:
        scale_min = scale_max
    if steps <= 1:
        return [max(scale_min, 0.01)]
    scales = np.linspace(scale_min, scale_max, steps)
    return [float(s) for s in scales if s > 0.01]


def resize_template(template: np.ndarray, scale: float) -> np.ndarray:
    """Resize template theo tỉ lệ scale."""
    h, w = template.shape[:2]
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    return cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)


def nms_boxes(boxes: List[BBox], scores: List[float], iou_threshold: float) -> List[int]:
    """Áp dụng NMS và trả về chỉ số bbox được giữ lại."""
    if not boxes:
        return []
    x1 = np.array([b[0] for b in boxes], dtype=np.float32)
    y1 = np.array([b[1] for b in boxes], dtype=np.float32)
    x2 = np.array([b[0] + b[2] for b in boxes], dtype=np.float32)
    y2 = np.array([b[1] + b[3] for b in boxes], dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32)

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores_np.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]
    return keep


def color_from_name(name: str) -> Tuple[int, int, int]:
    """Sinh màu ổn định dựa trên tên template."""
    seed = abs(hash(name)) % 255
    r = (seed * 97) % 255
    g = (seed * 57) % 255
    b = (seed * 17) % 255
    return int(b), int(g), int(r)


def draw_results(
    image_bgr: np.ndarray,
    results: Iterable[Dict],
    template_colors: Dict[str, Tuple[int, int, int]],
) -> np.ndarray:
    """Vẽ bbox và điểm cosine similarity (nếu có) lên ảnh."""
    output = image_bgr.copy()
    for item in results:
        x, y, w, h = item["bbox"]
        name = item["template_name"]
        color = template_colors.get(name, (0, 255, 0))
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        match_score = item.get("match_score")
        sim = item.get("cosine_similarity")
        label_parts = []
        if match_score is not None:
            label_parts.append(f"m:{match_score:.2f}")
        if sim is not None:
            label_parts.append(f"c:{sim:.2f}")
        if label_parts:
            label = " ".join(label_parts)
            cv2.putText(
                output,
                label,
                (x, max(y - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return output


def to_json(data: List[Dict]) -> str:
    """Chuyển danh sách kết quả sang chuỗi JSON."""
    return json.dumps(data, ensure_ascii=False, indent=2)


def is_mostly_white(image_bgr: np.ndarray, threshold: int = 100, min_foreground_ratio: float = 0.01) -> bool:
    """Kiem tra vung anh co qua it net ve (gan nhu toan trang)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    foreground = gray < threshold
    ratio = float(foreground.mean())
    return ratio < min_foreground_ratio
