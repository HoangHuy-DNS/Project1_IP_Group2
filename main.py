import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

from processing import (
    restore_image,
    compare_methods,
    process_batch
)

from evaluation import evaluate_image


# ==========================================================
# BIẾN TOÀN CỤC
# ==========================================================

original_image = None
result_image = None

reference_image = None
reference_path = None

image_path = None


# ==========================================================
# HIỂN THỊ ẢNH
# ==========================================================

def show_image(image, label, max_width=500, max_height=400):
    """
    Hiển thị ảnh OpenCV lên Label của Tkinter.
    """

    if image is None:
        return

    if len(image.shape) == 2:
        display_image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )
    else:
        display_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    pil_image = Image.fromarray(display_image)

    width, height = pil_image.size

    scale = min(
        max_width / width,
        max_height / height,
        1
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    pil_image = pil_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    photo = ImageTk.PhotoImage(pil_image)

    label.configure(
        image=photo,
        text=""
    )

    label.image = photo


# ==========================================================
# RESET ĐÁNH GIÁ
# ==========================================================

def reset_evaluation():
    """
    Đưa phần đánh giá PSNR/SSIM về trạng thái ban đầu.
    """

    psnr_value_label.config(
        text="PSNR: --"
    )

    ssim_value_label.config(
        text="SSIM: --"
    )


# ==========================================================
# MỞ ẢNH
# ==========================================================

def open_image():
    global original_image
    global result_image
    global image_path
    global reference_image
    global reference_path

    path = filedialog.askopenfilename(
        title="Chọn ảnh",
        filetypes=[
            (
                "Image files",
                "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"
            ),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Bitmap", "*.bmp"),
            ("TIFF", "*.tif *.tiff"),
            ("All files", "*.*")
        ]
    )

    if not path:
        return

    image = cv2.imread(path)

    if image is None:
        messagebox.showerror(
            "Lỗi",
            "Không thể đọc ảnh đã chọn."
        )
        return

    # ------------------------------------------------------
    # Cập nhật ảnh gốc
    # ------------------------------------------------------

    original_image = image
    result_image = None
    image_path = path

    # ------------------------------------------------------
    # Khi mở ảnh mới -> xóa ảnh tham chiếu cũ
    # ------------------------------------------------------

    reference_image = None
    reference_path = None

    # ------------------------------------------------------
    # Hiển thị ảnh gốc
    # ------------------------------------------------------

    show_image(
        original_image,
        original_image_label
    )

    # ------------------------------------------------------
    # Xóa ảnh kết quả cũ
    # ------------------------------------------------------

    result_image_label.configure(
        image="",
        text="Chưa có kết quả"
    )

    result_image_label.image = None

    # ------------------------------------------------------
    # Reset đánh giá
    # ------------------------------------------------------

    reset_evaluation()

    status_label.config(
        text=(
            f"Đã mở ảnh: "
            f"{os.path.basename(path)} | "
            f"Chưa chọn ảnh tham chiếu"
        )
    )


# ==========================================================
# CHỌN ẢNH THAM CHIẾU
# ==========================================================

def open_reference_image():
    """
    Chọn ảnh tham chiếu (Ground Truth)
    dùng để tính PSNR và SSIM.

    Ảnh tham chiếu phải là ảnh tương ứng
    với ảnh đang được xử lý.
    """

    global reference_image
    global reference_path

    if original_image is None:
        messagebox.showwarning(
            "Thông báo",
            "Vui lòng mở ảnh cần khôi phục trước."
        )
        return

    file_path = filedialog.askopenfilename(
        title="Chọn ảnh tham chiếu",
        filetypes=[
            (
                "Image files",
                "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"
            ),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Bitmap", "*.bmp"),
            ("TIFF", "*.tif *.tiff"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        return

    try:

        image = cv2.imread(file_path)

        if image is None:
            messagebox.showerror(
                "Lỗi",
                "Không thể đọc ảnh tham chiếu."
            )
            return

        reference_image = image
        reference_path = file_path

        # --------------------------------------------------
        # Thông báo nếu kích thước khác
        # --------------------------------------------------

        if original_image.shape[:2] != reference_image.shape[:2]:

            response = messagebox.askyesno(
                "Kích thước khác nhau",
                "Ảnh gốc và ảnh tham chiếu có kích thước khác nhau.\n\n"
                "Hệ thống sẽ tự điều chỉnh kích thước khi tính PSNR/SSIM.\n\n"
                "Bạn có muốn tiếp tục không?"
            )

            if not response:
                reference_image = None
                reference_path = None
                return

        # --------------------------------------------------
        # Nếu đã có kết quả -> tính lại đánh giá
        # --------------------------------------------------

        if result_image is not None:
            update_evaluation()

        status_label.config(
            text=(
                "Đã chọn ảnh tham chiếu: "
                f"{os.path.basename(file_path)}"
            )
        )

    except Exception as e:

        messagebox.showerror(
            "Lỗi",
            f"Không thể mở ảnh tham chiếu:\n{e}"
        )


# ==========================================================
# ĐÁNH GIÁ PSNR / SSIM
# ==========================================================

def update_evaluation():
    """
    Tính PSNR và SSIM giữa:
    
        Ảnh tham chiếu
             ↓
        Ảnh sau khôi phục
    """

    # ------------------------------------------------------
    # Chưa có ảnh kết quả
    # ------------------------------------------------------

    if result_image is None:

        psnr_value_label.config(
            text="PSNR: Chưa có kết quả"
        )

        ssim_value_label.config(
            text="SSIM: Chưa có kết quả"
        )

        return

    # ------------------------------------------------------
    # Chưa có ảnh tham chiếu
    # ------------------------------------------------------

    if reference_image is None:

        psnr_value_label.config(
            text="PSNR: Chưa có ảnh tham chiếu"
        )

        ssim_value_label.config(
            text="SSIM: Chưa có ảnh tham chiếu"
        )

        return

    try:

        metrics = evaluate_image(
            reference_image,
            result_image
        )

        psnr = metrics["psnr"]
        ssim = metrics["ssim"]

        # --------------------------------------------------
        # PSNR
        # --------------------------------------------------

        if np.isinf(psnr):
            psnr_text = "∞"
        else:
            psnr_text = f"{psnr:.2f} dB"

        # --------------------------------------------------
        # SSIM
        # --------------------------------------------------

        ssim_text = f"{ssim:.4f}"

        psnr_value_label.config(
            text=f"PSNR: {psnr_text}"
        )

        ssim_value_label.config(
            text=f"SSIM: {ssim_text}"
        )

    except Exception as e:

        psnr_value_label.config(
            text="PSNR: Lỗi"
        )

        ssim_value_label.config(
            text="SSIM: Lỗi"
        )

        print(
            f"Lỗi đánh giá PSNR/SSIM: {e}"
        )


# ==========================================================
# KHÔI PHỤC ẢNH
# ==========================================================

def restore_current_image():
    global result_image

    if original_image is None:

        messagebox.showwarning(
            "Thông báo",
            "Vui lòng mở ảnh trước."
        )

        return

    method = method_combo.get()

    try:

        kernel_size = int(
            kernel_combo.get()
        )

    except ValueError:

        messagebox.showerror(
            "Lỗi",
            "Kernel size không hợp lệ."
        )

        return

    try:

        # --------------------------------------------------
        # Khôi phục ảnh
        # --------------------------------------------------

        result_image = restore_image(
            original_image,
            method,
            kernel_size
        )

        if result_image is None:
            raise ValueError(
                "Không tạo được ảnh kết quả."
            )

        # --------------------------------------------------
        # Hiển thị kết quả
        # --------------------------------------------------

        show_image(
            result_image,
            result_image_label
        )

        # --------------------------------------------------
        # Đánh giá PSNR / SSIM
        # --------------------------------------------------

        update_evaluation()

        status_label.config(
            text=(
                f"Đã khôi phục bằng: {method} | "
                f"Kernel size: {kernel_size}"
            )
        )

    except Exception as e:

        messagebox.showerror(
            "Lỗi khôi phục ảnh",
            str(e)
        )


# ==========================================================
# SO SÁNH CÁC PHƯƠNG PHÁP
# ==========================================================

def compare_current_image():

    if original_image is None:

        messagebox.showwarning(
            "Thông báo",
            "Vui lòng mở ảnh trước."
        )

        return

    try:

        kernel_size = int(
            kernel_combo.get()
        )

    except ValueError:

        messagebox.showerror(
            "Lỗi",
            "Kernel size không hợp lệ."
        )

        return

    try:

        # --------------------------------------------------
        # Chạy các phương pháp
        # --------------------------------------------------

        results = compare_methods(
            original_image,
            kernel_size
        )

        if not results:

            messagebox.showwarning(
                "Thông báo",
                "Không có kết quả để so sánh."
            )

            return

        # --------------------------------------------------
        # Tạo cửa sổ so sánh
        # --------------------------------------------------

        compare_window = tk.Toplevel(root)

        compare_window.title(
            "So sánh các phương pháp"
        )

        compare_window.geometry(
            "1250x780"
        )

        compare_window.minsize(
            1100,
            650
        )

        # --------------------------------------------------
        # Tiêu đề
        # --------------------------------------------------

        title_label = tk.Label(
            compare_window,
            text="SO SÁNH CÁC PHƯƠNG PHÁP KHÔI PHỤC",
            font=("Arial", 16, "bold")
        )

        title_label.pack(
            pady=(10, 5)
        )

        # --------------------------------------------------
        # Thông tin kernel
        # --------------------------------------------------

        info_label = tk.Label(
            compare_window,
            text=f"Kernel size: {kernel_size}",
            font=("Arial", 10)
        )

        info_label.pack(
            pady=3
        )

        # --------------------------------------------------
        # Thông tin ảnh tham chiếu
        # --------------------------------------------------

        if reference_image is not None:

            reference_info = (
                "Đánh giá: Có ảnh tham chiếu"
            )

        else:

            reference_info = (
                "Đánh giá: Chưa có ảnh tham chiếu"
            )

        reference_info_label = tk.Label(
            compare_window,
            text=reference_info,
            font=("Arial", 10)
        )

        reference_info_label.pack(
            pady=3
        )

        # --------------------------------------------------
        # Khung chứa kết quả
        # --------------------------------------------------

        results_frame = tk.Frame(
            compare_window
        )

        results_frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        # --------------------------------------------------
        # Hiển thị từng phương pháp
        # --------------------------------------------------

        for column, (method_name, image) in enumerate(
            results.items()
        ):

            method_frame = tk.Frame(
                results_frame,
                relief="groove",
                borderwidth=1
            )

            method_frame.grid(
                row=0,
                column=column,
                padx=5,
                pady=5,
                sticky="n"
            )

            # ------------------------------------------------
            # Tên phương pháp
            # ------------------------------------------------

            method_label = tk.Label(
                method_frame,
                text=method_name,
                font=("Arial", 11, "bold")
            )

            method_label.pack(
                pady=5
            )

            # ------------------------------------------------
            # Ảnh kết quả
            # ------------------------------------------------

            image_label = tk.Label(
                method_frame,
                text="Đang hiển thị..."
            )

            image_label.pack(
                padx=5,
                pady=5
            )

            show_image(
                image,
                image_label,
                max_width=240,
                max_height=300
            )

            # ------------------------------------------------
            # PSNR / SSIM
            # ------------------------------------------------

            if reference_image is not None:

                try:

                    metrics = evaluate_image(
                        reference_image,
                        image
                    )

                    psnr = metrics["psnr"]
                    ssim = metrics["ssim"]

                    if np.isinf(psnr):
                        psnr_text = "∞"
                    else:
                        psnr_text = f"{psnr:.2f} dB"

                    ssim_text = f"{ssim:.4f}"

                    psnr_label = tk.Label(
                        method_frame,
                        text=f"PSNR: {psnr_text}",
                        font=("Arial", 10)
                    )

                    psnr_label.pack(
                        pady=(5, 2)
                    )

                    ssim_label = tk.Label(
                        method_frame,
                        text=f"SSIM: {ssim_text}",
                        font=("Arial", 10)
                    )

                    ssim_label.pack(
                        pady=(2, 8)
                    )

                except Exception:

                    metric_label = tk.Label(
                        method_frame,
                        text="PSNR/SSIM: Lỗi"
                    )

                    metric_label.pack(
                        pady=8
                    )

            else:

                metric_label = tk.Label(
                    method_frame,
                    text="PSNR: --\nSSIM: --",
                    font=("Arial", 10)
                )

                metric_label.pack(
                    pady=8
                )

        # --------------------------------------------------
        # Nút đóng
        # --------------------------------------------------

        close_button = tk.Button(
            compare_window,
            text="Đóng",
            width=12,
            command=compare_window.destroy
        )

        close_button.pack(
            pady=10
        )

    except Exception as e:

        messagebox.showerror(
            "Lỗi so sánh",
            str(e)
        )


# ==========================================================
# BATCH PROCESSING
# ==========================================================

def batch_processing():
    """
    Xử lý hàng loạt ảnh trong một thư mục.

    Phương pháp và kernel được lấy trực tiếp
    từ GUI.
    """

    input_folder = filedialog.askdirectory(
        title="Chọn thư mục ảnh đầu vào"
    )

    if not input_folder:
        return

    output_folder = filedialog.askdirectory(
        title="Chọn thư mục lưu kết quả"
    )

    if not output_folder:
        return

    method = method_combo.get()

    try:

        kernel_size = int(
            kernel_combo.get()
        )

    except ValueError:

        messagebox.showerror(
            "Lỗi",
            "Kernel size không hợp lệ."
        )

        return

    # ------------------------------------------------------
    # Xác nhận
    # ------------------------------------------------------

    confirm = messagebox.askyesno(
        "Xác nhận Batch Processing",
        f"Phương pháp: {method}\n"
        f"Kernel size: {kernel_size}\n\n"
        f"Thư mục đầu vào:\n{input_folder}\n\n"
        f"Thư mục đầu ra:\n{output_folder}\n\n"
        "Bạn có muốn bắt đầu xử lý không?"
    )

    if not confirm:
        return

    try:

        status_label.config(
            text="Đang xử lý Batch Processing..."
        )

        root.update_idletasks()

        # --------------------------------------------------
        # Xử lý Batch
        # --------------------------------------------------

        success, failed = process_batch(
            input_folder=input_folder,
            output_folder=output_folder,
            method=method,
            kernel_size=kernel_size
        )

        total = success + failed

        status_label.config(
            text=(
                f"Batch hoàn thành: "
                f"{success}/{total} ảnh thành công"
            )
        )

        messagebox.showinfo(
            "Batch Processing hoàn tất",
            f"Phương pháp: {method}\n"
            f"Kernel size: {kernel_size}\n\n"
            f"Tổng số ảnh: {total}\n"
            f"Thành công: {success}\n"
            f"Thất bại: {failed}"
        )

    except Exception as e:

        status_label.config(
            text="Batch Processing thất bại."
        )

        messagebox.showerror(
            "Lỗi Batch Processing",
            str(e)
        )


# ==========================================================
# LƯU KẾT QUẢ
# ==========================================================

def save_result():

    if result_image is None:

        messagebox.showwarning(
            "Thông báo",
            "Chưa có ảnh kết quả để lưu."
        )

        return

    save_path = filedialog.asksaveasfilename(
        title="Lưu ảnh kết quả",
        defaultextension=".jpg",
        filetypes=[
            ("JPEG", "*.jpg"),
            ("PNG", "*.png"),
            ("Bitmap", "*.bmp"),
            ("All files", "*.*")
        ]
    )

    if not save_path:
        return

    try:

        success = cv2.imwrite(
            save_path,
            result_image
        )

        if not success:

            raise ValueError(
                "Không thể lưu ảnh."
            )

        status_label.config(
            text=(
                f"Đã lưu: "
                f"{os.path.basename(save_path)}"
            )
        )

        messagebox.showinfo(
            "Thành công",
            "Đã lưu ảnh kết quả."
        )

    except Exception as e:

        messagebox.showerror(
            "Lỗi lưu ảnh",
            str(e)
        )


# ==========================================================
# TẠO GIAO DIỆN
# ==========================================================

root = tk.Tk()

root.title(
    "Ứng dụng khôi phục ảnh cũ - Project 1"
)

root.geometry(
    "1200x750"
)

root.minsize(
    1000,
    650
)


# ==========================================================
# TIÊU ĐỀ
# ==========================================================

title = tk.Label(
    root,
    text="ỨNG DỤNG KHÔI PHỤC ẢNH CŨ",
    font=("Arial", 20, "bold")
)

title.pack(
    pady=10
)


subtitle = tk.Label(
    root,
    text="Project 1 - Xử lý ảnh",
    font=("Arial", 11)
)

subtitle.pack(
    pady=(0, 10)
)


# ==========================================================
# KHUNG ĐIỀU KHIỂN
# ==========================================================

control_frame = tk.Frame(
    root,
    relief="groove",
    borderwidth=1
)

control_frame.pack(
    fill="x",
    padx=15,
    pady=10
)


# ==========================================================
# PHƯƠNG PHÁP
# ==========================================================

method_label = tk.Label(
    control_frame,
    text="Phương pháp:"
)

method_label.grid(
    row=0,
    column=0,
    padx=10,
    pady=10
)


method_combo = ttk.Combobox(
    control_frame,
    state="readonly",
    width=20,
    values=[
        "Median Blur",
        "Gaussian Blur",
        "Bilateral Filter",
        "Sharpening",
        "Inpainting"
    ]
)

method_combo.current(0)

method_combo.grid(
    row=0,
    column=1,
    padx=10,
    pady=10
)


# ==========================================================
# KERNEL SIZE
# ==========================================================

kernel_label = tk.Label(
    control_frame,
    text="Kernel size:"
)

kernel_label.grid(
    row=0,
    column=2,
    padx=10,
    pady=10
)


kernel_combo = ttk.Combobox(
    control_frame,
    state="readonly",
    width=10,
    values=[
        "3",
        "5",
        "7",
        "9"
    ]
)

kernel_combo.set("5")

kernel_combo.grid(
    row=0,
    column=3,
    padx=10,
    pady=10
)


# ==========================================================
# NÚT MỞ ẢNH
# ==========================================================

open_button = tk.Button(
    control_frame,
    text="Mở ảnh",
    width=14,
    command=open_image
)

open_button.grid(
    row=0,
    column=4,
    padx=5,
    pady=10
)


# ==========================================================
# NÚT CHỌN ẢNH THAM CHIẾU
# ==========================================================

reference_button = tk.Button(
    control_frame,
    text="Chọn ảnh tham chiếu",
    width=18,
    command=open_reference_image
)

reference_button.grid(
    row=0,
    column=5,
    padx=5,
    pady=10
)


# ==========================================================
# NÚT KHÔI PHỤC
# ==========================================================

restore_button = tk.Button(
    control_frame,
    text="Khôi phục",
    width=14,
    command=restore_current_image
)

restore_button.grid(
    row=0,
    column=6,
    padx=5,
    pady=10
)


# ==========================================================
# NÚT SO SÁNH
# ==========================================================

compare_button = tk.Button(
    control_frame,
    text="So sánh",
    width=14,
    command=compare_current_image
)

compare_button.grid(
    row=0,
    column=7,
    padx=5,
    pady=10
)


# ==========================================================
# NÚT BATCH PROCESSING
# ==========================================================

batch_button = tk.Button(
    control_frame,
    text="Batch Processing",
    width=16,
    command=batch_processing
)

batch_button.grid(
    row=1,
    column=4,
    padx=5,
    pady=8
)


# ==========================================================
# NÚT LƯU KẾT QUẢ
# ==========================================================

save_button = tk.Button(
    control_frame,
    text="Lưu kết quả",
    width=14,
    command=save_result
)

save_button.grid(
    row=1,
    column=5,
    padx=5,
    pady=8
)


# ==========================================================
# KHUNG HIỂN THỊ ẢNH
# ==========================================================

image_frame = tk.Frame(
    root
)

image_frame.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=10
)


# ==========================================================
# ẢNH GỐC
# ==========================================================

original_frame = tk.LabelFrame(
    image_frame,
    text="Ảnh gốc",
    font=("Arial", 11, "bold")
)

original_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=5
)


original_image_label = tk.Label(
    original_frame,
    text="Chưa mở ảnh",
    font=("Arial", 12)
)

original_image_label.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


# ==========================================================
# ẢNH KẾT QUẢ
# ==========================================================

result_frame = tk.LabelFrame(
    image_frame,
    text="Ảnh sau khôi phục",
    font=("Arial", 11, "bold")
)

result_frame.pack(
    side="right",
    fill="both",
    expand=True,
    padx=5
)


result_image_label = tk.Label(
    result_frame,
    text="Chưa có kết quả",
    font=("Arial", 12)
)

result_image_label.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


# ==========================================================
# KHUNG ĐÁNH GIÁ
# ==========================================================

evaluation_frame = tk.Frame(
    root,
    relief="groove",
    borderwidth=1
)

evaluation_frame.pack(
    fill="x",
    padx=15,
    pady=5
)


evaluation_title = tk.Label(
    evaluation_frame,
    text="ĐÁNH GIÁ KẾT QUẢ",
    font=("Arial", 11, "bold")
)

evaluation_title.pack(
    side="left",
    padx=15,
    pady=8
)


psnr_value_label = tk.Label(
    evaluation_frame,
    text="PSNR: --",
    font=("Arial", 11)
)

psnr_value_label.pack(
    side="left",
    padx=20
)


ssim_value_label = tk.Label(
    evaluation_frame,
    text="SSIM: --",
    font=("Arial", 11)
)

ssim_value_label.pack(
    side="left",
    padx=20
)


# ==========================================================
# STATUS
# ==========================================================

status_label = tk.Label(
    root,
    text="Sẵn sàng.",
    anchor="w",
    relief="sunken"
)

status_label.pack(
    fill="x",
    side="bottom"
)


# ==========================================================
# CHẠY CHƯƠNG TRÌNH
# ==========================================================

root.mainloop()