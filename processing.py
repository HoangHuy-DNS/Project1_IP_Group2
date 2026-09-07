import cv2
import numpy as np


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
# 4. HÀM KHÔI PHỤC ẢNH CHÍNH
# ==========================================================

def restore_image(image, method="Median Blur", kernel_size=5):
    """
    Khôi phục ảnh theo phương pháp được lựa chọn.

    Parameters
    ----------
    image : numpy.ndarray
        Ảnh đầu vào ở dạng BGR.

    method : str
        Phương pháp xử lý.

    kernel_size : int
        Kích thước kernel.

    Returns
    -------
    numpy.ndarray
        Ảnh sau khi xử lý.
    """

    # Kiểm tra ảnh
    if image is None:
        raise ValueError("Ảnh đầu vào không hợp lệ.")

    # Kiểm tra kernel
    kernel_size = int(kernel_size)

    if kernel_size < 3:
        kernel_size = 3

    # Kernel phải là số lẻ
    if kernel_size % 2 == 0:
        kernel_size += 1

    # ------------------------------------------------------
    # Median Blur
    # ------------------------------------------------------

    if method == "Median Blur":

        return median_blur(
            image,
            kernel_size
        )

    # ------------------------------------------------------
    # Gaussian Blur
    # ------------------------------------------------------

    elif method == "Gaussian Blur":

        return gaussian_blur(
            image,
            kernel_size
        )

    # ------------------------------------------------------
    # Bilateral Filter
    # ------------------------------------------------------

    elif method == "Bilateral Filter":

        return bilateral_filter(
            image,
            kernel_size
        )

    # ------------------------------------------------------
    # Phương pháp không hợp lệ
    # ------------------------------------------------------

    else:

        raise ValueError(
            f"Không tìm thấy phương pháp: {method}"
        )


# ==========================================================
# 5. PSNR
# ==========================================================

def calculate_psnr(original, restored):
    """
    Tính PSNR giữa ảnh gốc và ảnh sau khôi phục.

    PSNR càng lớn thì ảnh khôi phục càng gần ảnh tham chiếu.
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

    mse = np.mean(
        (original - restored) ** 2
    )

    # Hai ảnh giống hoàn toàn
    if mse == 0:
        return float("inf")

    max_pixel = 255.0

    psnr = 10 * np.log10(
        (max_pixel ** 2) / mse
    )

    return float(psnr)


# ==========================================================
# 6. SSIM
# ==========================================================

def calculate_ssim(original, restored):
    """
    Tính SSIM giữa hai ảnh.

    Giá trị SSIM nằm gần khoảng:
        1.0  -> rất giống nhau
        0.0  -> ít tương đồng
    """

    if original is None or restored is None:
        return 0.0

    # Nếu kích thước khác nhau
    if original.shape[:2] != restored.shape[:2]:

        restored = cv2.resize(
            restored,
            (original.shape[1], original.shape[0])
        )

    # Chuyển sang grayscale
    if len(original.shape) == 3:
        gray_original = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray_original = original

    if len(restored.shape) == 3:
        gray_restored = cv2.cvtColor(
            restored,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray_restored = restored

    # Chuyển sang float
    gray_original = gray_original.astype(np.float64)
    gray_restored = gray_restored.astype(np.float64)

    # Các hằng số của SSIM
    L = 255.0

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # Gaussian window
    mu_original = cv2.GaussianBlur(
        gray_original,
        (11, 11),
        1.5
    )

    mu_restored = cv2.GaussianBlur(
        gray_restored,
        (11, 11),
        1.5
    )

    mu_original_sq = mu_original ** 2
    mu_restored_sq = mu_restored ** 2
    mu_original_restored = (
        mu_original * mu_restored
    )

    sigma_original_sq = cv2.GaussianBlur(
        gray_original ** 2,
        (11, 11),
        1.5
    ) - mu_original_sq

    sigma_restored_sq = cv2.GaussianBlur(
        gray_restored ** 2,
        (11, 11),
        1.5
    ) - mu_restored_sq

    sigma_original_restored = (
        cv2.GaussianBlur(
            gray_original * gray_restored,
            (11, 11),
            1.5
        )
        - mu_original_restored
    )

    # Công thức SSIM
    numerator = (
        (2 * mu_original_restored + C1)
        *
        (2 * sigma_original_restored + C2)
    )

    denominator = (
        (mu_original_sq + mu_restored_sq + C1)
        *
        (sigma_original_sq + sigma_restored_sq + C2)
    )

    ssim_map = numerator / (denominator + 1e-12)

    return float(np.mean(ssim_map))


# ==========================================================
# 7. SO SÁNH NHIỀU PHƯƠNG PHÁP
# ==========================================================

def compare_methods(image, kernel_size=5):
    """
    Chạy tất cả phương pháp trên cùng một ảnh.

    Returns
    -------
    dict
        Dictionary chứa kết quả của từng phương pháp.
    """

    methods = [
        "Median Blur",
        "Gaussian Blur",
        "Bilateral Filter"
    ]

    results = {}

    for method in methods:

        result = restore_image(
            image,
            method,
            kernel_size
        )

        results[method] = result

    return results


# ==========================================================
# 8. XỬ LÝ BATCH
# ==========================================================

def process_batch(
    input_folder,
    output_folder,
    method="Median Blur",
    kernel_size=5
):
    """
    Xử lý toàn bộ ảnh trong một thư mục.

    Returns
    -------
    tuple
        (số lượng thành công, số lượng lỗi)
    """

    import os

    # Tạo thư mục output
    os.makedirs(
        output_folder,
        exist_ok=True
    )

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

    for filename in os.listdir(input_folder):

        if not filename.lower().endswith(
            supported_extensions
        ):
            continue

        input_path = os.path.join(
            input_folder,
            filename
        )

        output_path = os.path.join(
            output_folder,
            filename
        )

        image = cv2.imread(input_path)

        if image is None:
            failed += 1
            continue

        try:

            result = restore_image(
                image,
                method,
                kernel_size
            )

            cv2.imwrite(
                output_path,
                result
            )

            success += 1

        except Exception:
            failed += 1

    return success, failed