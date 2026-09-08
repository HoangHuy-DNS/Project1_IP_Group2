# ỨNG DỤNG KHÔI PHỤC ẢNH CŨ

## 1. Giới thiệu

Đây là đồ án môn Xử lý ảnh với chủ đề:

**Ứng dụng khôi phục ảnh cũ**

Mục tiêu của dự án là xây dựng một ứng dụng có giao diện đồ họa (GUI) hỗ trợ khôi phục ảnh bằng các phương pháp xử lý ảnh, cho phép người dùng thay đổi tham số, xem kết quả trước và sau khi xử lý, lưu ảnh kết quả và xử lý nhiều ảnh.

## 2. Thành viên nhóm

- Nguyễn Hoàng Huy
- Nguyễn Đăng Khoa
- Nguyễn Viết Anh Khôi

## 3. Công nghệ sử dụng

- Python
- OpenCV
- NumPy
- Pillow
- Tkinter
- Visual Studio Code

## 4. Chức năng hiện tại

- Mở ảnh
- Hiển thị ảnh ban đầu
- Khôi phục ảnh
- Lựa chọn phương pháp xử lý
- Điều chỉnh Kernel
- Hiển thị ảnh kết quả
- So sánh các phương pháp xử lý
- Batch Processing
- Lưu ảnh kết quả
- Tính PSNR
- Tính SSIM

## 5. Phương pháp xử lý

Các phương pháp đang được cài đặt trong chương trình:

- Median Blur
- Gaussian Blur
- Bilateral Filter
- Sharpening
- Inpainting

## 6. Đánh giá kết quả

Dự án sử dụng hai chỉ số để đánh giá chất lượng ảnh:

### PSNR

PSNR (Peak Signal-to-Noise Ratio) được sử dụng để đánh giá mức độ sai khác giữa ảnh tham chiếu và ảnh sau khi xử lý.

Giá trị PSNR càng cao thì mức độ tương đồng giữa hai ảnh càng lớn.

### SSIM

SSIM (Structural Similarity Index Measure) được sử dụng để đánh giá mức độ tương đồng về cấu trúc và thông tin hình ảnh.

Giá trị SSIM càng gần 1 thì hai ảnh càng tương đồng.

## 7. Batch Processing

Chức năng Batch Processing cho phép xử lý nhiều ảnh trong một thư mục.

Người dùng có thể:

- Chọn thư mục ảnh đầu vào.
- Chọn phương pháp xử lý.
- Điều chỉnh Kernel.
- Chọn thư mục lưu kết quả.
- Xử lý nhiều ảnh tự động.

## 8. Dataset

Dataset của dự án hiện có:

- 82 file ảnh.
- 81 ảnh đọc được bằng OpenCV.
- 1 file ảnh không đọc được.

Dataset gồm các định dạng:

- JPG
- JPEG
- PNG

Số lượng ảnh hợp lệ hiện tại đáp ứng yêu cầu tối thiểu 50 ảnh cho Project 1.

## 9. Cấu trúc dự án

```text
Project1_IP_Group2/
│
├── main.py
├── processing.py
├── evaluation.py
├── batch.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│
├── data/
│   ├── input/
│   ├── output/
│   └── reference/
│
└── results/
```
