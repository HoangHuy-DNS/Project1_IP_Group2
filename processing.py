import cv2
import numpy as np
from pathlib import Path


# ============================================================
# 1. MEDIAN BLUR
# ============================================================
def median_blur(image, kernel_size=5):
    """
    Giảm nhiễu bằng Median Blur.
    Phù hợp với nhiễu dạng điểm / salt-and-pepper.
    """
    return cv2.medianBlur(image, kernel_size)


# ============================================================
# 2. GAUSSIAN BLUR
# ============================================================
def gaussian_blur(image, kernel_size=5, sigma=0):
    """
    Làm mượt ảnh và giảm nhiễu bằng Gaussian Blur.
    """
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigma
    )


# ============================================================
# 3. BILATERAL FILTER
# ============================================================
def bilateral_filter(image, kernel_size=5):
    """
    Giảm nhiễu nhưng cố gắng giữ các đường biên của ảnh.
    """
    return cv2.bilateralFilter(
        image,
        kernel_size,
        75,
        75
    )


# ============================================================
# 4. SHARPENING
# ============================================================
def sharpen_image(image, kernel_size=5):
    """
    Làm tăng độ sắc nét của ảnh.

    kernel_size được sử dụng để điều chỉnh mức độ
    làm mượt trước khi tăng độ sắc nét.
    """

    # Đảm bảo kernel là số lẻ và >= 3
    kernel_size = max(3, int(kernel_size))

    if kernel_size % 2 == 0:
        kernel_size += 1

    # Làm mượt ảnh theo kernel được chọn
    blurred = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )

    # Unsharp Mask:
    # ảnh sắc nét = ảnh gốc + phần chi tiết
    sharpened = cv2.addWeighted(
        image,
        1.5,
        blurred,
        -0.5,
        0
    )

    return sharpened


# ============================================================
# 5. INPAINTING
# ============================================================
def inpaint_image(image, kernel_size=5):
    """
    Khôi phục các vùng nhỏ bị lỗi, đốm hoặc vết xước
    bằng phương pháp Inpainting.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Đảm bảo kernel hợp lệ
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Phát hiện các vùng tối bất thường
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    # Ngưỡng thấp hơn để phát hiện các lỗi nhỏ
    _, mask = cv2.threshold(
        blackhat,
        15,
        255,
        cv2.THRESH_BINARY
    )

    # Loại bỏ các vùng nhiễu rất nhỏ
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    # Làm rộng nhẹ vùng cần khôi phục
    mask = cv2.dilate(
        mask,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    # Inpainting
    result = cv2.inpaint(
        image,
        mask,
        3,
        cv2.INPAINT_TELEA
    )

    return result


# ============================================================
# 6. RESTORE IMAGE
# ============================================================
def restore_image(image, method="Median Blur", kernel_size=5):
    """
    Khôi phục ảnh theo phương pháp được lựa chọn.
    """

    if image is None:
        return None

    # Đảm bảo kernel hợp lệ
    kernel_size = max(3, int(kernel_size))

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
        raise ValueError(
            f"Phương pháp xử lý không được hỗ trợ: {method}"
        )


# ============================================================
# 7. PSNR
# ============================================================
def calculate_psnr(original, restored):
    """
    Tính chỉ số PSNR giữa ảnh gốc và ảnh khôi phục.
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

    return float(
        10 * np.log10(
            (max_pixel ** 2) / mse
        )
    )


# ============================================================
# 8. SSIM
# ============================================================
def calculate_ssim(original, restored):
    """
    Tính chỉ số SSIM giữa ảnh gốc và ảnh khôi phục.
    """

    if original is None or restored is None:
        return 0.0

    # Điều chỉnh kích thước
    if original.shape[:2] != restored.shape[:2]:
        restored = cv2.resize(
            restored,
            (original.shape[1], original.shape[0])
        )

    # Chuyển sang ảnh xám
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2GRAY
        )
    else:
        original_gray = original.copy()

    if len(restored.shape) == 3:
        restored_gray = cv2.cvtColor(
            restored,
            cv2.COLOR_BGR2GRAY
        )
    else:
        restored_gray = restored.copy()

    original_gray = original_gray.astype(np.float64)
    restored_gray = restored_gray.astype(np.float64)

    # Hằng số SSIM
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Trung bình cục bộ
    mu_original = cv2.GaussianBlur(
        original_gray,
        (11, 11),
        1.5
    )

    mu_restored = cv2.GaussianBlur(
        restored_gray,
        (11, 11),
        1.5
    )

    mu_original_sq = mu_original ** 2
    mu_restored_sq = mu_restored ** 2
    mu_original_restored = mu_original * mu_restored

    # Phương sai
    sigma_original_sq = (
        cv2.GaussianBlur(
            original_gray ** 2,
            (11, 11),
            1.5
        )
        - mu_original_sq
    )

    sigma_restored_sq = (
        cv2.GaussianBlur(
            restored_gray ** 2,
            (11, 11),
            1.5
        )
        - mu_restored_sq
    )

    # Hiệp phương sai
    sigma_original_restored = (
        cv2.GaussianBlur(
            original_gray * restored_gray,
            (11, 11),
            1.5
        )
        - mu_original_restored
    )

    # Công thức SSIM
    numerator = (
        (2 * mu_original_restored + C1)
        * (2 * sigma_original_restored + C2)
    )

    denominator = (
        (mu_original_sq + mu_restored_sq + C1)
        * (sigma_original_sq + sigma_restored_sq + C2)
    )

    ssim_map = numerator / denominator

    return float(np.mean(ssim_map))


# ============================================================
# 9. SO SÁNH CÁC PHƯƠNG PHÁP
# ============================================================
def compare_methods(image, kernel_size=5):
    """
    Chạy nhiều phương pháp xử lý trên cùng một ảnh.
    """

    methods = [
        "Median Blur",
        "Gaussian Blur",
        "Bilateral Filter",
        "Sharpening"
    ]

    results = {}

    for method in methods:
        results[method] = restore_image(
            image,
            method,
            kernel_size
        )

    return results


# ============================================================
# 10. BATCH PROCESSING
# ============================================================
def process_batch(
    input_folder,
    output_folder,
    method="Sharpening",
    kernel_size=5
):
    """
    Xử lý nhiều ảnh trong một thư mục.
    """

    input_path = Path(input_folder)
    output_path = Path(output_folder)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    success = 0
    failed = 0

    for file_path in input_path.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported_extensions:
            continue

        try:
            image = cv2.imread(str(file_path))

            if image is None:
                failed += 1
                continue

            result = restore_image(
                image,
                method,
                kernel_size
            )

            # Giữ cấu trúc thư mục tương đối
            relative_path = file_path.relative_to(
                input_path
            )

            output_file = output_path / relative_path

            output_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            if cv2.imwrite(
                str(output_file),
                result
            ):
                success += 1
            else:
                failed += 1

        except Exception:
            failed += 1

    return success, failed