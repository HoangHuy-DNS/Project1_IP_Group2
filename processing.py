import cv2
import numpy as np
from pathlib import Path


# ==========================================================
# 1. MEDIAN BLUR
# ==========================================================

def median_blur(image, kernel_size=5):
    """
    Lọc Median để giảm nhiễu muối tiêu.
    """
    return cv2.medianBlur(image, kernel_size)


# ==========================================================
# 2. GAUSSIAN BLUR
# ==========================================================

def gaussian_blur(image, kernel_size=5, sigma=0):
    """
    Lọc Gaussian để làm mượt và giảm nhiễu.
    """
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigma
    )


# ==========================================================
# 3. BILATERAL FILTER
# ==========================================================

def bilateral_filter(image, kernel_size=5):
    """
    Lọc Bilateral giúp giảm nhiễu nhưng giữ biên tốt hơn.
    """
    return cv2.bilateralFilter(
        image,
        kernel_size,
        75,
        75
    )


# ==========================================================
# 4. SHARPENING (Tăng độ nét, chống mờ nhòe)
# ==========================================================

def sharpen_image(image, kernel_size=5):
    """
    Bộ lọc tăng cường độ nét giúp ảnh không bị mờ nhòe.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


# ==========================================================
# 5. INPAINTING (Xử lý đốm, vết bẩn, trầy xước)
# ==========================================================

def inpaint_image(image, kernel_size=5):
    """
    Thuật toán vá ảnh để xóa đốm đen hoặc vết bẩn.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


# ==========================================================
# 6. HÀM KHÔI PHỤC ẢNH CHÍNH
# ==========================================================

def restore_image(image, method="Median Blur", kernel_size=5):
    """
    Khôi phục ảnh theo phương pháp được lựa chọn.
    """
    if image is None:
        raise ValueError("Ảnh đầu vào không hợp lệ.")

    kernel_size = int(kernel_size)
    if kernel_size < 3:
        kernel_size = 3

    if kernel_size % 2 == 0:
        kernel_size += 1

    if method == "Median Blur":
        return median_blur(image, kernel_size)
    elif method == "Gaussian Blur":
        return gaussian_blur(image, kernel_size)
    elif method == "Bilateral Filter":
        return bilateral_filter(image, kernel_size)
    elif method == "Sharpening":
        return sharpen_image(image, kernel_size)
    elif method == "Inpainting":
        return inpaint_image(image, kernel_size)
    else:
        raise ValueError(f"Không tìm thấy phương pháp: {method}")


# ==========================================================
# 7. PSNR
# ==========================================================

def calculate_psnr(original, restored):
    """
    Tính PSNR giữa ảnh gốc và ảnh sau khôi phục.
    """
    if original is None or restored is None:
        return 0.0

    if original.shape != restored.shape:
        restored = cv2.resize(
            restored,
            (original.shape[1], original.shape[0])
        )

    original = original.astype(np.float64)
    restored = restored.astype(np.float64)

    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return float("inf")

    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return float(psnr)


# ==========================================================
# 8. SSIM
# ==========================================================

def calculate_ssim(original, restored):
    """
    Tính SSIM giữa hai ảnh.
    """
    if original is None or restored is None:
        return 0.0

    if original.shape[:2] != restored.shape[:2]:
        restored = cv2.resize(
            restored,
            (original.shape[1], original.shape[0])
        )

    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
    gray_restored = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY) if len(restored.shape) == 3 else restored

    gray_original = gray_original.astype(np.float64)
    gray_restored = gray_restored.astype(np.float64)

    L = 255.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    mu_original = cv2.GaussianBlur(gray_original, (11, 11), 1.5)
    mu_restored = cv2.GaussianBlur(gray_restored, (11, 11), 1.5)

    mu_original_sq = mu_original ** 2
    mu_restored_sq = mu_restored ** 2
    mu_original_restored = mu_original * mu_restored

    sigma_original_sq = cv2.GaussianBlur(gray_original ** 2, (11, 11), 1.5) - mu_original_sq
    sigma_restored_sq = cv2.GaussianBlur(gray_restored ** 2, (11, 11), 1.5) - mu_restored_sq
    sigma_original_restored = cv2.GaussianBlur(gray_original * gray_restored, (11, 11), 1.5) - mu_original_restored

    numerator = (2 * mu_original_restored + C1) * (2 * sigma_original_restored + C2)
    denominator = (mu_original_sq + mu_restored_sq + C1) * (sigma_original_sq + sigma_restored_sq + C2)

    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


# ==========================================================
# 9. SO SÁNH NHIỀU PHƯƠNG PHÁP
# ==========================================================

def compare_methods(image, kernel_size=5):
    """
    Chạy các phương pháp chính trên cùng một ảnh.
    """
    methods = [
        "Median Blur",
        "Gaussian Blur",
        "Bilateral Filter",
        "Sharpening"
    ]

    results = {}
    for method in methods:
        results[method] = restore_image(image, method, kernel_size)

    return results


# ==========================================================
# 10. XỬ LÝ BATCH 
# ==========================================================

def process_batch(
    input_folder,
    output_folder,
    method="Sharpening",
    kernel_size=5
):
    """
    Xử lý toàn bộ ảnh trong thư mục và các thư mục con.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    supported_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    )

    success = 0
    failed = 0

    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            relative_path = file_path.relative_to(input_path)
            target_file_path = output_path / relative_path
            target_file_path.parent.mkdir(parents=True, exist_ok=True)

            image = cv2.imread(str(file_path))
            if image is None:
                failed += 1
                continue

            try:
                result = restore_image(image, method, kernel_size)
                cv2.imwrite(str(target_file_path), result)
                success += 1
            except Exception:
                failed += 1

    return success, failed