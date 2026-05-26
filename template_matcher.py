from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from utils import (
    BBox,
    compute_scale_candidates,
    nms_boxes,
    resize_template,
    rotate_image,
    trim_white_border,
)


MATCH_METHODS = {
    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
}


@dataclass
class MatchResult:
    """Kết quả khớp cho một template tại một vị trí cụ thể."""
    template_name: str
    bbox: BBox
    match_score: float
    angle: float
    scale: float
    template_variant: np.ndarray
    cosine_similarity: Optional[float] = None

    def to_dict(self) -> Dict:
        """Chuyển kết quả sang dict để xuất JSON."""
        data = asdict(self)
        data.pop("template_variant", None)
        return data


class TemplateMatcher:
    """Thực hiện template matching đa tỉ lệ, đa góc xoay và NMS."""
    def __init__(
        self,
        match_threshold: float = 0.7,
        iou_threshold: float = 0.3,
        match_method: str = "TM_CCOEFF_NORMED",
        angles: Optional[Sequence[float]] = None,
        scale_min: float = 0.1,
        scale_max: float = 2.0,
        scale_steps: int = 10,
        max_detections_per_template: int = 50,
        use_multithreading: bool = True,
        use_gray: bool = True,
    ) -> None:
        """Khởi tạo matcher với các ngưỡng và cấu hình tìm kiếm."""
        if match_method not in MATCH_METHODS:
            raise ValueError(f"Phương pháp matching không hỗ trợ: {match_method}")
        self.match_threshold = match_threshold
        self.iou_threshold = iou_threshold
        self.match_method_name = match_method
        self.match_method = MATCH_METHODS[match_method]
        self.angles = list(angles) if angles is not None else [0, 45, 90, 135, 180, 225, 270, 315]
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_steps = scale_steps
        self.max_detections_per_template = max_detections_per_template
        self.use_multithreading = use_multithreading
        self.use_gray = use_gray

    def _match_single_template(
        self,
        image_bgr: np.ndarray,
        template_bgr: np.ndarray,
        template_name: str,
    ) -> List[MatchResult]:
        """Khớp một template đơn lẻ với ảnh đầu vào."""
        image_h, image_w = image_bgr.shape[:2]
        image_match = (
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if self.use_gray else image_bgr
        )
        template_bgr = trim_white_border(template_bgr)
        t_h, t_w = template_bgr.shape[:2]
        min_image_dim = float(min(image_h, image_w))
        min_template_dim = float(min(t_h, t_w))
        if min_template_dim > 0 and min_image_dim > 0:
            base_scale = (0.1 * min_image_dim) / min_template_dim
            template_bgr = resize_template(template_bgr, base_scale)
        results: List[MatchResult] = []
        for angle in self.angles:
            rotated = rotate_image(template_bgr, angle)
            scales = compute_scale_candidates(
                image_bgr.shape, rotated.shape, self.scale_min, self.scale_max, self.scale_steps
            )
            for scale in scales:
                scaled = resize_template(rotated, scale)
                t_h, t_w = scaled.shape[:2]
                if t_h < 5 or t_w < 5:
                    continue
                if t_h > image_h or t_w > image_w:
                    continue
                template_match = (
                    cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY) if self.use_gray else scaled
                )
                match_map = cv2.matchTemplate(image_match, template_match, self.match_method)
                loc = np.where(match_map >= self.match_threshold)
                if loc[0].size == 0:
                    continue
                scores = match_map[loc]
                order = np.argsort(scores)[::-1]
                top_k = order[: self.max_detections_per_template]
                for idx in top_k:
                    y = int(loc[0][idx])
                    x = int(loc[1][idx])
                    score = float(scores[idx])
                    results.append(
                        MatchResult(
                            template_name=template_name,
                            bbox=(x, y, t_w, t_h),
                            match_score=score,
                            angle=angle,
                            scale=scale,
                            template_variant=scaled,
                        )
                    )
        if not results:
            return []
        boxes = [r.bbox for r in results]
        scores = [r.match_score for r in results]
        keep = nms_boxes(boxes, scores, self.iou_threshold)
        filtered = [results[i] for i in keep][: self.max_detections_per_template]
        return filtered

    def find(
        self,
        image_bgr: np.ndarray,
        templates: Dict[str, np.ndarray],
    ) -> List[MatchResult]:
        """Khớp nhiều template và trả về danh sách kết quả."""
        if not templates:
            return []
        items = list(templates.items())
        if self.use_multithreading and len(items) > 1:
            with ThreadPoolExecutor(max_workers=min(8, len(items))) as executor:
                futures = [
                    executor.submit(self._match_single_template, image_bgr, t, name)
                    for name, t in items
                ]
                results = [r for f in futures for r in f.result()]
        else:
            results = []
            for name, tmpl in items:
                results.extend(self._match_single_template(image_bgr, tmpl, name))
        return results
