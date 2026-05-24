from pathlib import Path
from glob import glob
import argparse
import gc
import os

import torch

from template_matcher import (
    ImageDataset,
    CreateModel,
)
from utils import build_feature_extractor, nms_multi, plot_result_multi, run_multi_sample


def parse_args() -> argparse.Namespace:
    """
    Phân tích các đối số dòng lệnh.
    
    Returns:
        argparse.Namespace: Đối tượng chứa các tham số được truyền vào từ dòng lệnh.
    """
    parser = argparse.ArgumentParser(description='CNN Template Matching Implementation')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-s', '--sample_image', default='sample/sample1.jpg')
    parser.add_argument('-t', '--template_images_dir', default='template/')
    parser.add_argument('-ss', '--sample_images_dir')
    parser.add_argument('-r', '--result_images_dir', default='result/')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--thresh', type=float, default=0.7)
    parser.add_argument('--model', type=str, default='convnext_tiny', choices=['convnext_tiny', 'efficientnet_b4', 'mobilenet_v3'], help='Backbone model trích xuất đặc trưng')
    return parser.parse_args()


def main() -> None:
    """
    Hàm chính điều khiển luồng thực thi của ứng dụng.
    Xử lý theo chế độ một ảnh (single image) hoặc xử lý hàng loạt (batch).
    """
    args = parse_args()

    # Các đường dẫn đầu vào/ra được tách riêng để dễ chỉnh sửa khi chạy lệnh.
    template_dir = args.template_images_dir
    result_path = args.result_images_dir
    os.makedirs(result_path, exist_ok=True)

    # Nếu user yêu cầu sử dụng CUDA nhưng máy không có GPU, tự động chuyển về dùng CPU.
    use_cuda = args.cuda and torch.cuda.is_available()
    if args.cuda and not use_cuda:
        print('CUDA was requested but is not available. Falling back to CPU.')

    print(f'Đang định nghĩa mô hình (define model: {args.model})...')
    # Backbone được chọn là nơi trích xuất đặc trưng (feature) của ảnh template và ảnh sample.
    model = CreateModel(
        model=build_feature_extractor(model_name=args.model, pretrained=True),
        alpha=args.alpha,
        use_cuda=use_cuda,
    )

    if not args.sample_images_dir:
        # Chế độ xử lý 1 ảnh: so khớp tất cả template trong thư mục template/ với 1 ảnh sample.
        print('Chế độ 1 ảnh (One Sample Image Is Inputted)')
        image_path = args.sample_image
        dataset = ImageDataset(Path(template_dir), image_path, thresh=args.thresh)
        print('Đang tính toán điểm số (calculate score)...')
        # Vòng lặp: trích xuất feature -> chuẩn hóa -> tính confidence map -> chuyển sang điểm số.
        scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
        print('Đang thực hiện nms (non-maximum suppression)...')
        # Loại bỏ các bounding box trùng lặp bằng thuật toán non-maximum suppression (NMS).
        boxes, indices, confidences = nms_multi(scores, w_array, h_array, thresh_list)
        _ = plot_result_multi(dataset.image_raw, boxes, indices, show=False, save_name='result.png')
        print('Đã lưu file result.png')
        return

    # Chế độ xử lý theo lô (batch): xử lý lần lượt từng ảnh trong thư mục sample_images_dir.
    print('Chế độ nhiều ảnh (Image Directory Is Inputted)')
    sample_images_dir = args.sample_images_dir
    images = glob(os.path.join(sample_images_dir, '*'))
    for index, image in enumerate(images, start=1):
        print('-----', index, '/', len(images), '-----')
        image_name = Path(image).stem
        print(f'Ảnh Sample: {image_name} đang được xử lý (Processing)...')
        dataset = ImageDataset(Path(template_dir), image, thresh=args.thresh)
        print('Đang tính toán điểm số (calculate score)...')
        # Mỗi template sẽ sinh ra một score map, sau đó gộp chung lại thông qua NMS và vẽ kết quả.
        scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
        print('Đang thực hiện nms (non-maximum suppression)...')
        boxes, indices, confidences = nms_multi(scores, w_array, h_array, thresh_list)
        _ = plot_result_multi(dataset.image_raw, boxes, indices, show=True, save_name=os.path.join(result_path, image_name) + '.png')
        print('Đã lưu ảnh kết quả (result image was saved)')
        del dataset
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()