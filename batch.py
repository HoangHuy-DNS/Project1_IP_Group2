from pathlib import Path
from processing import process_batch

if __name__ == "__main__":
    # Tự động lấy thư mục gốc chứa file batch.py
    BASE_DIR = Path(__file__).resolve().parent
    
    input_dir = BASE_DIR / "dataset"
    output_dir = BASE_DIR / "dataset_processed"

    print(f"[*] Thư mục gốc của project: {BASE_DIR}")
    print(f"[*] Thư mục đầu vào: {input_dir}")
    print(f"[*] Thư mục đầu ra: {output_dir}\n")

    # Gọi hàm xử lý hàng loạt và nhận về số lượng thành công, thất bại
    success, failed = process_batch(
        input_folder=str(input_dir),
        output_folder=str(output_dir),
        method="Sharpening",
        kernel_size=5
    )

    # Tính tổng số ảnh
    total_images = success + failed

    # In bảng thống kê tổng kết
    print("\n" + "="*40)
    print("         BẢNG THỐNG KÊ KẾT QUẢ")
    print("="*40)
    print(f" Tổng số ảnh quét được  : {total_images}")
    print(f" Xử lý thành công       : {success}")
    print(f" Xử lý thất bại         : {failed}")
    print("="*40)