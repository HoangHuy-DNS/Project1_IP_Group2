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
# 4. SHARPENING (CẢI TIẾN)
# ============================================================

def sharpen_image(image, kernel_size=5, amount=1.5):
    """
    Làm tăng độ sắc nét của ảnh bằng Unsharp Masking chuẩn.

    Parameters
    ----------
    image : numpy.ndarray
        Ảnh đầu vào (BGR/Gray).
    kernel_size : int
        Kích thước kernel làm mờ (nên là 5 hoặc 7).
    amount : float
        Mức độ làm nét (1.0 là vừa, 1.5 - 2.5 là nét mạnh).
    """
    if image is None:
        return None

    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    # 1. Ép sang float32 để tính toán chính xác không bị tràn số (overflow)
    img_float = image.astype(np.float32)

    # 2. Tạo bản mờ với sigmaX rõ rệt (2.0 giúp tách biệt biên cạnh rõ hơn)
    blurred = cv2.GaussianBlur(
        img_float,
        (kernel_size, kernel_size),
        sigmaX=2.0
    )

    # 3. Công thức Unsharp Masking chuẩn:
    # Sharpened = Original + Amount * (Original - Blurred)
    # Hay: Original * (1 + Amount) - Blurred * Amount
    sharpened_float = cv2.addWeighted(
        img_float,
        1.0 + amount,
        blurred,
        -amount,
        0
    )

    # 4. Giới hạn dải pixel trong [0, 255] và chuyển lại uint8
    sharpened = np.clip(sharpened_float, 0, 255).astype(np.uint8)

    return sharpened

# ============================================================
# 5. INPAINTING
# ============================================================

def inpaint_image(image, kernel_size=5):
    """
    Khôi phục các vùng nhỏ bị lỗi, đốm hoặc vết xước
    bằng phương pháp Inpainting.
    """

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

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

    # Phát hiện vùng lỗi
    _, mask = cv2.threshold(
        blackhat,
        15,
        255,
        cv2.THRESH_BINARY
    )

    # Loại bỏ vùng nhiễu rất nhỏ
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

def restore_image(
    image,
    method="Median Blur",
    kernel_size=5
):
    """
    Khôi phục ảnh theo phương pháp được lựa chọn.
    """

    if image is None:
        return None

    kernel_size = max(3, int(kernel_size))

    if kernel_size % 2 == 0:
        kernel_size += 1

    if method == "Median Blur":
        return median_blur(
            image,
            kernel_size
        )

    elif method == "Gaussian Blur":
        return gaussian_blur(
            image,
            kernel_size
        )

    elif method == "Bilateral Filter":
        return bilateral_filter(
            image,
            kernel_size
        )

    elif method == "Sharpening":
        # Truyền thêm amount=1.5 (hoặc 2.0 nếu muốn nét cực mạnh để test)
        return sharpen_image(
            image,
            kernel_size,
            amount=1.5
        )

    elif method == "Inpainting":
        return inpaint_image(
            image,
            kernel_size
        )

    else:
        raise ValueError(
            f"Phương pháp xử lý không được hỗ trợ: {method}"
        )


# ============================================================
# 7. SO SÁNH CÁC PHƯƠNG PHÁP
# ============================================================

def compare_methods(image, kernel_size=5):
    """
    Chạy tất cả các phương pháp xử lý trên cùng một ảnh.

    Returns
    -------
    dict
        Dictionary chứa ảnh kết quả của từng phương pháp.
    """

    methods = [
        "Median Blur",
        "Gaussian Blur",
        "Bilateral Filter",
        "Sharpening",
        "Inpainting"
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
# 8. BATCH PROCESSING
# ============================================================

def process_batch(
    input_folder,
    output_folder,
    method="Sharpening",
    kernel_size=5
):
    """
    Xử lý nhiều ảnh trong một thư mục.

    Parameters
    ----------
    input_folder : str
        Thư mục chứa ảnh đầu vào.

    output_folder : str
        Thư mục lưu ảnh kết quả.

    method : str
        Phương pháp xử lý.

    kernel_size : int
        Kích thước kernel.

    Returns
    -------
    tuple
        (success, failed)
    """

    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        return 0, 0

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
            image = cv2.imread(
                str(file_path)
            )

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

            output_file = (
                output_path / relative_path
            )

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