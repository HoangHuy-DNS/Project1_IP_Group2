from pathlib import Path

from processing import process_batch


if __name__ == "__main__":

    # ========================================================
    # Xác định thư mục gốc của project
    # ========================================================

    BASE_DIR = Path(__file__).resolve().parent

    # Dataset đầu vào
    input_dir = BASE_DIR / "data" / "dataset"

    # Kết quả Batch Processing
    output_dir = BASE_DIR / "data" / "dataset_processed"

    print(
        f"[*] Thư mục gốc của project: {BASE_DIR}"
    )

    print(
        f"[*] Thư mục đầu vào: {input_dir}"
    )

    print(
        f"[*] Thư mục đầu ra: {output_dir}\n"
    )

    # ========================================================
    # Kiểm tra dataset
    # ========================================================

    if not input_dir.exists():
        print(
            "[!] Không tìm thấy thư mục dataset."
        )
        print(
            f"[!] Đường dẫn: {input_dir}"
        )
        raise SystemExit(1)

    # ========================================================
    # Batch Processing
    # ========================================================

    success, failed = process_batch(
        input_folder=str(input_dir),
        output_folder=str(output_dir),
        method="Sharpening",
        kernel_size=5
    )

    # ========================================================
    # Thống kê
    # ========================================================

    total_images = success + failed

    print("\n" + "=" * 40)
    print("         BẢNG THỐNG KÊ KẾT QUẢ")
    print("=" * 40)

    print(
        f" Tổng số ảnh quét được  : {total_images}"
    )

    print(
        f" Xử lý thành công       : {success}"
    )

    print(
        f" Xử lý thất bại         : {failed}"
    )

    print("=" * 40)