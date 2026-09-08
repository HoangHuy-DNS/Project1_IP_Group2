import cv2
import numpy as np


# ==========================================================
# 1. PSNR
# ==========================================================

def calculate_psnr(reference, restored):
    """
    Tính PSNR giữa ảnh tham chiếu và ảnh khôi phục.

    Parameters
    ----------
    reference : numpy.ndarray
        Ảnh tham chiếu (Ground Truth).

    restored : numpy.ndarray
        Ảnh sau khi khôi phục.

    Returns
    -------
    float
        Giá trị PSNR tính theo dB.
    """

    if reference is None or restored is None:
        raise ValueError("Ảnh tham chiếu hoặc ảnh khôi phục không hợp lệ.")

    if reference.shape != restored.shape:
        restored = cv2.resize(
            restored,
            (reference.shape[1], reference.shape[0])
        )

    reference = reference.astype(np.float64)
    restored = restored.astype(np.float64)

    mse = np.mean(
        (reference - restored) ** 2
    )

    if mse == 0:
        return float("inf")

    max_pixel = 255.0

    psnr = 10 * np.log10(
        (max_pixel ** 2) / mse
    )

    return float(psnr)


# ==========================================================
# 2. SSIM
# ==========================================================

def calculate_ssim(reference, restored):
    """
    Tính SSIM giữa ảnh tham chiếu và ảnh khôi phục.

    Parameters
    ----------
    reference : numpy.ndarray
        Ảnh tham chiếu (Ground Truth).

    restored : numpy.ndarray
        Ảnh sau khi khôi phục.

    Returns
    -------
    float
        Giá trị SSIM.
    """

    if reference is None or restored is None:
        raise ValueError("Ảnh tham chiếu hoặc ảnh khôi phục không hợp lệ.")

    # Chuyển sang ảnh xám
    if len(reference.shape) == 3:
        reference_gray = cv2.cvtColor(
            reference,
            cv2.COLOR_BGR2GRAY
        )
    else:
        reference_gray = reference.copy()

    if len(restored.shape) == 3:
        restored_gray = cv2.cvtColor(
            restored,
            cv2.COLOR_BGR2GRAY
        )
    else:
        restored_gray = restored.copy()

    # Đảm bảo hai ảnh cùng kích thước
    if reference_gray.shape != restored_gray.shape:
        restored_gray = cv2.resize(
            restored_gray,
            (
                reference_gray.shape[1],
                reference_gray.shape[0]
            )
        )

    reference_gray = reference_gray.astype(np.float64)
    restored_gray = restored_gray.astype(np.float64)

    # Các hằng số của SSIM
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Gaussian window
    mu_reference = cv2.GaussianBlur(
        reference_gray,
        (11, 11),
        1.5
    )

    mu_restored = cv2.GaussianBlur(
        restored_gray,
        (11, 11),
        1.5
    )

    mu_reference_sq = mu_reference ** 2
    mu_restored_sq = mu_restored ** 2
    mu_reference_restored = (
        mu_reference * mu_restored
    )

    sigma_reference_sq = (
        cv2.GaussianBlur(
            reference_gray ** 2,
            (11, 11),
            1.5
        )
        - mu_reference_sq
    )

    sigma_restored_sq = (
        cv2.GaussianBlur(
            restored_gray ** 2,
            (11, 11),
            1.5
        )
        - mu_restored_sq
    )

    sigma_reference_restored = (
        cv2.GaussianBlur(
            reference_gray * restored_gray,
            (11, 11),
            1.5
        )
        - mu_reference_restored
    )

    numerator = (
        (2 * mu_reference_restored + C1)
        * (2 * sigma_reference_restored + C2)
    )

    denominator = (
        (mu_reference_sq + mu_restored_sq + C1)
        * (sigma_reference_sq + sigma_restored_sq + C2)
    )

    ssim_map = numerator / denominator

    return float(np.mean(ssim_map))


# ==========================================================
# 3. ĐÁNH GIÁ ẢNH
# ==========================================================

def evaluate_image(reference, restored):
    """
    Tính đồng thời PSNR và SSIM.

    Returns
    -------
    dict
        {
            "psnr": ...,
            "ssim": ...
        }
    """

    psnr = calculate_psnr(
        reference,
        restored
    )

    ssim = calculate_ssim(
        reference,
        restored
    )

    return {
        "psnr": psnr,
        "ssim": ssim
    }