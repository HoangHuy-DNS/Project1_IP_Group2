# ỨNG DỤNG KHÔI PHỤC ẢNH CŨ

## 1. Giới thiệu

Đây là đồ án môn Xử lý ảnh với chủ đề:

**Ứng dụng khôi phục ảnh cũ**

Mục tiêu của dự án là xây dựng một ứng dụng có giao diện đồ họa (GUI) hỗ trợ khôi phục ảnh bằng các phương pháp xử lý ảnh.

Ứng dụng cho phép người dùng lựa chọn phương pháp xử lý, thay đổi tham số, xem ảnh trước và sau khi xử lý, đánh giá kết quả bằng PSNR và SSIM, so sánh các phương pháp và xử lý nhiều ảnh bằng Batch Processing.

Project 1 của nhóm sử dụng **dataset ảnh**, không sử dụng video.

---

## 2. Thành viên nhóm

- Nguyễn Hoàng Huy
- Nguyễn Đăng Khoa
- Nguyễn Viết Anh Khôi

---

## 3. Công nghệ sử dụng

- Python
- OpenCV
- NumPy
- Pillow
- Tkinter
- Visual Studio Code

---

## 4. Chức năng

Ứng dụng cung cấp các chức năng:

- Mở ảnh
- Hiển thị ảnh ban đầu
- Lựa chọn phương pháp xử lý
- Điều chỉnh Kernel
- Khôi phục ảnh
- Hiển thị ảnh trước và sau khi xử lý
- Tính PSNR
- Tính SSIM
- So sánh các phương pháp xử lý
- Batch Processing
- Lưu ảnh kết quả

---

## 5. Các phương pháp xử lý

Các phương pháp được cài đặt:

### 5.1. Median Blur

Sử dụng bộ lọc trung vị để giảm nhiễu, đặc biệt phù hợp với nhiễu dạng điểm hoặc salt-and-pepper.

### 5.2. Gaussian Blur

Làm mượt ảnh và giảm nhiễu bằng bộ lọc Gaussian.

### 5.3. Bilateral Filter

Giảm nhiễu trong khi cố gắng giữ lại các đường biên của ảnh.

### 5.4. Sharpening

Tăng độ sắc nét của ảnh bằng phương pháp Unsharp Mask.

### 5.5. Inpainting

Phát hiện các vùng lỗi nhỏ như đốm hoặc vết xước và khôi phục vùng ảnh đó bằng phương pháp Inpainting.

---

## 6. Đánh giá kết quả

Ứng dụng sử dụng hai chỉ số:

### PSNR

PSNR (Peak Signal-to-Noise Ratio) được sử dụng để định lượng mức độ sai khác giữa hai ảnh.

Giá trị PSNR càng cao thì mức độ tương đồng giữa hai ảnh càng lớn.

### SSIM

SSIM (Structural Similarity Index Measure) đánh giá mức độ tương đồng về cấu trúc và thông tin hình ảnh.

Giá trị SSIM càng gần 1 thì hai ảnh càng tương đồng.

Trong phiên bản hiện tại, PSNR và SSIM được sử dụng để định lượng mức độ tương đồng giữa ảnh đầu vào và ảnh sau xử lý, kết hợp với đánh giá trực quan khi so sánh kết quả.

---

## 7. So sánh các phương pháp

Chức năng So sánh cho phép chạy nhiều phương pháp trên cùng một ảnh với cùng tham số Kernel.

Các phương pháp được so sánh:

- Median Blur
- Gaussian Blur
- Bilateral Filter
- Sharpening
- Inpainting

Kết quả của từng phương pháp được hiển thị cùng với các chỉ số PSNR và SSIM.

Việc so sánh giúp đánh giá sự khác biệt giữa các phương pháp trên từng ảnh cụ thể.

---

## 8. Batch Processing

Batch Processing cho phép xử lý nhiều ảnh trong cùng một thư mục.

Các bước thực hiện:

1. Chọn thư mục ảnh đầu vào.
2. Chọn phương pháp xử lý.
3. Điều chỉnh Kernel.
4. Chọn thư mục lưu kết quả.
5. Tiến hành xử lý hàng loạt.

Chương trình hỗ trợ các định dạng:

- JPG
- JPEG
- PNG
- BMP

Kết quả Batch Processing được lưu trong thư mục:

`data/dataset_processed/`

---

## 9. Dataset

Dataset hiện tại của Project 1 gồm:

- **214 file ảnh**
- Đáp ứng yêu cầu tối thiểu **≥50 ảnh** của đề bài.

Các định dạng ảnh trong dataset gồm:

- JPG
- JPEG
- PNG

Dataset được lưu tại:

`data/dataset/`

---

## 10. Cấu trúc dự án

```text
Project1_IP_Group2/
│
├── data/
│   ├── dataset/
│   │   └── 214 ảnh
│   │
│   ├── dataset_processed/
│   │   └── kết quả Batch Processing
│   │
│   ├── input/
│   └── output/
│
├── main.py
├── processing.py
├── evaluation.py
├── batch.py
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```
