import tkinter as tk

from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from PIL import Image, ImageTk

import cv2
import numpy as np
import os

from processing import (
    restore_image,
    calculate_psnr,
    calculate_ssim,
    compare_methods,
    process_batch
)


# ==========================================================
# BIẾN TOÀN CỤC
# ==========================================================

original_image = None
result_image = None
image_path = None


# ==========================================================
# HÀM HIỂN THỊ ẢNH
# ==========================================================

def show_image(image, label):
    """
    Hiển thị ảnh PIL lên Label.
    """

    if image is None:
        return

    display_image = image.copy()

    # Kích thước tối đa
    max_width = 700
    max_height = 570

    display_image.thumbnail(
        (max_width, max_height),
        Image.Resampling.LANCZOS
    )

    photo = ImageTk.PhotoImage(
        display_image
    )

    label.config(
        image=photo,
        text=""
    )

    # Giữ tham chiếu
    label.image = photo


# ==========================================================
# HÀM MỞ ẢNH
# ==========================================================

def open_image():
    """
    Mở ảnh từ máy tính.
    """

    global original_image
    global result_image
    global image_path

    path = filedialog.askopenfilename(
        title="Chọn ảnh",
        filetypes=[
            (
                "Image files",
                "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"
            ),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("BMP", "*.bmp"),
            ("TIFF", "*.tif *.tiff")
        ]
    )

    if not path:
        return

    try:

        image = Image.open(path)

        image = image.convert("RGB")

    except Exception as e:

        messagebox.showerror(
            "Lỗi",
            f"Không thể mở ảnh:\n{e}"
        )

        return

    # Lưu dữ liệu
    original_image = image
    image_path = path
    result_image = None

    # Hiển thị ảnh gốc
    show_image(
        original_image,
        original_label
    )

    # Xóa kết quả cũ
    result_label.config(
        image="",
        text="Chưa có kết quả"
    )

    result_label.image = None

    # Reset thông tin
    psnr_value.set("PSNR: --")
    ssim_value.set("SSIM: --")
    status_value.set(
        f"Đã mở: {os.path.basename(path)}"
    )


# ==========================================================
# HÀM KHÔI PHỤC ẢNH
# ==========================================================

def restore_current_image():
    """
    Khôi phục ảnh hiện tại.
    """

    global result_image

    if original_image is None:

        messagebox.showwarning(
            "Thông báo",
            "Vui lòng mở ảnh trước!"
        )

        return

    try:

        # Lấy phương pháp
        method = method_combo.get()

        # Lấy kernel
        kernel_size = int(
            kernel_combo.get()
        )

        # PIL → NumPy
        image = np.array(
            original_image
        )

        # RGB → BGR
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        # Xử lý
        result = restore_image(
            image,
            method,
            kernel_size
        )

        # BGR → RGB
        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )

        # NumPy → PIL
        result_image = Image.fromarray(
            result
        )

        # Hiển thị
        show_image(
            result_image,
            result_label
        )

        # --------------------------------------------------
        # Tính PSNR và SSIM
        #
        # Lưu ý:
        # Đây là phép so sánh với ảnh đầu vào.
        # Không phải ground-truth của ảnh cũ.
        # --------------------------------------------------

        original_cv = cv2.cvtColor(
            np.array(original_image),
            cv2.COLOR_RGB2BGR
        )

        psnr = calculate_psnr(
            original_cv,
            result
        )

        ssim = calculate_ssim(
            original_cv,
            result
        )

        if np.isinf(psnr):
            psnr_text = "PSNR: ∞"
        else:
            psnr_text = f"PSNR: {psnr:.2f} dB"

        ssim_text = f"SSIM: {ssim:.4f}"

        psnr_value.set(
            psnr_text
        )

        ssim_value.set(
            ssim_text
        )

        status_value.set(
            f"Đã xử lý bằng {method} - Kernel {kernel_size}"
        )

    except Exception as e:

        messagebox.showerror(
            "Lỗi xử lý",
            f"Không thể khôi phục ảnh:\n{e}"
        )


# ==========================================================
# HÀM LƯU KẾT QUẢ
# ==========================================================

def save_result():

    if result_image is None:

        messagebox.showwarning(
            "Thông báo",
            "Chưa có ảnh kết quả để lưu!"
        )

        return

    path = filedialog.asksaveasfilename(
        title="Lưu ảnh kết quả",
        defaultextension=".jpg",
        filetypes=[
            ("JPEG", "*.jpg"),
            ("PNG", "*.png"),
            ("BMP", "*.bmp"),
            ("TIFF", "*.tiff")
        ]
    )

    if not path:
        return

    try:

        result_image.save(path)

        messagebox.showinfo(
            "Thành công",
            "Đã lưu ảnh kết quả!"
        )

        status_value.set(
            f"Đã lưu: {os.path.basename(path)}"
        )

    except Exception as e:

        messagebox.showerror(
            "Lỗi",
            f"Không thể lưu ảnh:\n{e}"
        )


# ==========================================================
# HÀM SO SÁNH PHƯƠNG PHÁP
# ==========================================================

def compare_current_image():

    if original_image is None:

        messagebox.showwarning(
            "Thông báo",
            "Vui lòng mở ảnh trước!"
        )

        return

    try:

        kernel_size = int(
            kernel_combo.get()
        )

        image = np.array(
            original_image
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        results = compare_methods(
            image,
            kernel_size
        )

        # Tạo cửa sổ mới
        compare_window = tk.Toplevel(
            root
        )

        compare_window.title(
            "So sánh các phương pháp"
        )

        compare_window.geometry(
            "1400x800"
        )

        title = tk.Label(
            compare_window,
            text="SO SÁNH PHƯƠNG PHÁP KHÔI PHỤC",
            font=("Arial", 20, "bold")
        )

        title.pack(
            pady=15
        )

        image_container = tk.Frame(
            compare_window
        )

        image_container.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        # Giữ tham chiếu PhotoImage
        compare_photos = []

        column = 0

        for method, result in results.items():

            frame = tk.LabelFrame(
                image_container,
                text=method,
                font=("Arial", 11)
            )

            frame.grid(
                row=0,
                column=column,
                padx=10,
                pady=10,
                sticky="nsew"
            )

            image_rgb = cv2.cvtColor(
                result,
                cv2.COLOR_BGR2RGB
            )

            pil_image = Image.fromarray(
                image_rgb
            )

            pil_image.thumbnail(
                (400, 550),
                Image.Resampling.LANCZOS
            )

            photo = ImageTk.PhotoImage(
                pil_image
            )

            compare_photos.append(
                photo
            )

            label = tk.Label(
                frame,
                image=photo
            )

            label.pack(
                padx=10,
                pady=10
            )

            column += 1

        for i in range(3):

            image_container.columnconfigure(
                i,
                weight=1
            )

        # Giữ tham chiếu
        compare_window.compare_photos = compare_photos

    except Exception as e:

        messagebox.showerror(
            "Lỗi",
            f"Không thể so sánh:\n{e}"
        )


# ==========================================================
# HÀM BATCH PROCESSING
# ==========================================================

def batch_processing():

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

    try:

        method = method_combo.get()

        kernel_size = int(
            kernel_combo.get()
        )

        success, failed = process_batch(
            input_folder,
            output_folder,
            method,
            kernel_size
        )

        messagebox.showinfo(
            "Batch Processing",
            f"Đã xử lý xong!\n\n"
            f"Thành công: {success} ảnh\n"
            f"Lỗi: {failed} ảnh"
        )

        status_value.set(
            f"Batch: {success} thành công, {failed} lỗi"
        )

    except Exception as e:

        messagebox.showerror(
            "Lỗi Batch Processing",
            f"Không thể xử lý:\n{e}"
        )


# ==========================================================
# TẠO CỬA SỔ
# ==========================================================

root = tk.Tk()

root.title(
    "Ứng dụng khôi phục ảnh cũ"
)

root.geometry(
    "1650x950"
)

root.minsize(
    1100,
    700
)


# ==========================================================
# TITLE
# ==========================================================

title_label = tk.Label(
    root,
    text="ỨNG DỤNG KHÔI PHỤC ẢNH CŨ",
    font=("Arial", 26, "bold")
)

title_label.pack(
    pady=(15, 10)
)


# ==========================================================
# NÚT MỞ ẢNH
# ==========================================================

open_button = tk.Button(
    root,
    text="Mở ảnh",
    font=("Arial", 12),
    width=12,
    command=open_image
)

open_button.pack(
    pady=5
)


# ==========================================================
# KHU VỰC HIỂN THỊ ẢNH
# ==========================================================

image_frame = tk.Frame(
    root
)

image_frame.pack(
    fill="both",
    expand=True,
    padx=25,
    pady=10
)

# Cho hai cột bằng nhau
image_frame.columnconfigure(
    0,
    weight=1
)

image_frame.columnconfigure(
    1,
    weight=1
)

image_frame.rowconfigure(
    0,
    weight=1
)


# ==========================================================
# ẢNH BAN ĐẦU
# ==========================================================

original_frame = tk.LabelFrame(
    image_frame,
    text="Ảnh ban đầu",
    font=("Arial", 11)
)

original_frame.grid(
    row=0,
    column=0,
    padx=10,
    sticky="nsew"
)

original_label = tk.Label(
    original_frame,
    text="Chưa có ảnh",
    font=("Arial", 12)
)

original_label.pack(
    fill="both",
    expand=True
)


# ==========================================================
# ẢNH KẾT QUẢ
# ==========================================================

result_frame = tk.LabelFrame(
    image_frame,
    text="Ảnh kết quả",
    font=("Arial", 11)
)

result_frame.grid(
    row=0,
    column=1,
    padx=10,
    sticky="nsew"
)

result_label = tk.Label(
    result_frame,
    text="Chưa có kết quả",
    font=("Arial", 12)
)

result_label.pack(
    fill="both",
    expand=True
)


# ==========================================================
# KHU VỰC ĐIỀU KHIỂN
# ==========================================================

control_frame = tk.Frame(
    root
)

control_frame.pack(
    fill="x",
    padx=30,
    pady=5
)


# ==========================================================
# PHƯƠNG PHÁP
# ==========================================================

method_label = tk.Label(
    control_frame,
    text="Phương pháp:",
    font=("Arial", 11)
)

method_label.pack(
    side="left",
    padx=5
)

method_combo = ttk.Combobox(
    control_frame,
    values=[
        "Median Blur",
        "Gaussian Blur",
        "Bilateral Filter"
    ],
    state="readonly",
    width=18
)

method_combo.current(0)

method_combo.pack(
    side="left",
    padx=5
)


# ==========================================================
# KERNEL
# ==========================================================

kernel_label = tk.Label(
    control_frame,
    text="Kernel:",
    font=("Arial", 11)
)

kernel_label.pack(
    side="left",
    padx=(15, 5)
)

kernel_combo = ttk.Combobox(
    control_frame,
    values=[
        3,
        5,
        7,
        9
    ],
    state="readonly",
    width=5
)

kernel_combo.current(1)

kernel_combo.pack(
    side="left",
    padx=5
)


# ==========================================================
# NÚT KHÔI PHỤC
# ==========================================================

restore_button = tk.Button(
    control_frame,
    text="Khôi phục",
    font=("Arial", 11),
    width=12,
    command=restore_current_image
)

restore_button.pack(
    side="left",
    padx=10
)


# ==========================================================
# NÚT SO SÁNH
# ==========================================================

compare_button = tk.Button(
    control_frame,
    text="So sánh",
    font=("Arial", 11),
    width=10,
    command=compare_current_image
)

compare_button.pack(
    side="left",
    padx=5
)


# ==========================================================
# NÚT BATCH
# ==========================================================

batch_button = tk.Button(
    control_frame,
    text="Batch Processing",
    font=("Arial", 11),
    width=16,
    command=batch_processing
)

batch_button.pack(
    side="left",
    padx=5
)


# ==========================================================
# NÚT LƯU
# ==========================================================

save_button = tk.Button(
    control_frame,
    text="Lưu kết quả",
    font=("Arial", 11),
    width=13,
    command=save_result
)

save_button.pack(
    side="left",
    padx=5
)


# ==========================================================
# KHU VỰC ĐÁNH GIÁ
# ==========================================================

evaluation_frame = tk.Frame(
    root
)

evaluation_frame.pack(
    fill="x",
    padx=30,
    pady=5
)


psnr_value = tk.StringVar(
    value="PSNR: --"
)

ssim_value = tk.StringVar(
    value="SSIM: --"
)


psnr_label = tk.Label(
    evaluation_frame,
    textvariable=psnr_value,
    font=("Arial", 11, "bold")
)

psnr_label.pack(
    side="left",
    padx=15
)


ssim_label = tk.Label(
    evaluation_frame,
    textvariable=ssim_value,
    font=("Arial", 11, "bold")
)

ssim_label.pack(
    side="left",
    padx=15
)


# ==========================================================
# STATUS
# ==========================================================

status_value = tk.StringVar(
    value="Sẵn sàng"
)

status_label = tk.Label(
    root,
    textvariable=status_value,
    anchor="w",
    font=("Arial", 10)
)

status_label.pack(
    fill="x",
    padx=30,
    pady=(0, 10)
)


# ==========================================================
# CHẠY CHƯƠNG TRÌNH
# ==========================================================

root.mainloop()