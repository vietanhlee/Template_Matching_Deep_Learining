# CNN Template Matching (QATM-based)

Dự án này thực hiện kỹ thuật Template Matching nâng cao sử dụng mạng CNN (ConvNeXt, EfficientNet, MobileNet) để phát hiện và định vị các mẫu (templates) trên ảnh lớn (sample image), với khả năng tự động xoay template ở nhiều góc độ (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°) giúp vượt qua rào cản của phương pháp truyền thống.

## Cài đặt (Installation)

1. Cài đặt Python (khuyên dùng Python 3.8 trở lên).
2. Tạo môi trường ảo (tùy chọn nhưng khuyến khích):
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```
3. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```
*(Lưu ý: Nếu cần dùng GPU, hãy cài đặt phiên bản PyTorch hỗ trợ CUDA phù hợp với phần cứng của bạn tại [trang chủ PyTorch](https://pytorch.org/).)*

## Cấu hình và Chạy (Usage)

### 1. Giao diện Web (Gradio)
Đây là cách sử dụng trực quan và dễ dàng nhất.
```bash
python app.py
```
- Truy cập vào đường dẫn hiển thị trên terminal (thường là `http://127.0.0.1:7860`).
- Giao diện cho phép tải ảnh mẫu (sample) và tải lên nhiều ảnh template cùng lúc.
- Cung cấp tính năng chỉnh sửa tham số Alpha, Threshold, Scale và chọn Backbone Model trực tiếp.

### 2. Chạy qua Command Line (CLI)
Sử dụng file `main.py` để chạy thuật toán từ terminal, thích hợp cho việc viết script tự động hóa hoặc xử lý số lượng lớn (batch processing).

**Chạy với 1 ảnh mẫu (Single Image):**
```bash
python main.py -s sample/sample1.jpg -t template/ --model efficientnet_b4
```
Kết quả sẽ được lưu mặc định vào file `result.png`.

**Chạy với nhiều ảnh mẫu trong một thư mục (Batch Processing):**
```bash
python main.py -ss sample/ -t template/ -r result/ --model efficientnet_b4
```

**Các tham số cấu hình chính** (dùng `python main.py --help` để xem chi tiết):
- `--cuda`: Thêm cờ này nếu hệ thống có GPU NVIDIA và bạn muốn xử lý trên GPU.
- `-s` hoặc `--sample_image`: Đường dẫn trực tiếp đến ảnh mẫu (ví dụ: `sample.png`).
- `-t` hoặc `--template_images_dir`: Thư mục chứa các file ảnh template cần tìm.
- `-ss` hoặc `--sample_images_dir`: Thư mục chứa danh sách nhiều ảnh mẫu (dành cho batch).
- `-r` hoặc `--result_images_dir`: Thư mục để xuất lưu kết quả.
- `--alpha`: Tham số điều chỉnh độ gắt của Softmax (Mặc định: 25).
- `--thresh`: Ngưỡng NMS tương đối để lọc (Mặc định: 0.7).
- `--model`: Tên mô hình CNN trích xuất đặc trưng (`convnext_tiny`, `efficientnet_b4`, `mobilenet_v3`).
