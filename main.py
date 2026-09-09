import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from processing import (
    restore_image,
    compare_methods,
    process_batch
)

from evaluation import evaluate_image


# ============================================================
# BIẾN TOÀN CỤC
# ============================================================

original_image = None
result_image = None
image_path = None

original_photo = None
result_photo = None


# ============================================================
# HÀM HIỂN THỊ ẢNH
# ============================================================

def show_image(
    image,
    label,
    max_width=450,
    max_height=450
):
    """
    Hiển thị ảnh OpenCV lên Tkinter Label.
    """

    if image is None:
        label.config(
            image="",
            text="Không có ảnh"
        )
        return None

    try:
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

        pil_image = Image.fromarray(
            display_image
        )

        width, height = pil_image.size

        scale = min(
            max_width / width,
            max_height / height,
            1
        )

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        pil_image = pil_image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(
            pil_image
        )

        label.config(
            image=photo,
            text=""
        )

        label.image = photo

        return photo

    except Exception as error:
        label.config(
            image="",
            text=f"Lỗi hiển thị:\n{error}"
        )

        return None


# ============================================================
# LẤY KERNEL
# ============================================================

def get_kernel_size():
    """
    Lấy kernel từ Spinbox và đảm bảo kernel hợp lệ.
    """

    try:
        kernel_size = int(
            kernel_var.get()
        )
    except ValueError:
        kernel_size = 5

    kernel_size = max(
        3,
        kernel_size
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel_var.set(
        kernel_size
    )

    return kernel_size


# ============================================================
# CẬP NHẬT THÔNG TIN ẢNH
# ============================================================

def update_image_info():
    """
    Hiển thị thông tin ảnh đang mở.
    """

    if original_image is None:
        image_info_label.config(
            text="Chưa mở ảnh"
        )
        return

    height, width = (
        original_image.shape[:2]
    )

    channels = (
        original_image.shape[2]
        if len(original_image.shape) == 3
        else 1
    )

    image_info_label.config(
        text=(
            f"Kích thước: {width} x {height} | "
            f"Channels: {channels}"
        )
    )


# ============================================================
# MỞ ẢNH
# ============================================================

def open_image():
    global original_image
    global result_image
    global image_path

    file_path = filedialog.askopenfilename(
        title="Chọn ảnh",
        filetypes=[
            (
                "Image files",
                "*.jpg *.jpeg *.png *.bmp"
            ),
            ("JPG", "*.jpg"),
            ("JPEG", "*.jpeg"),
            ("PNG", "*.png"),
            ("BMP", "*.bmp")
        ]
    )

    if not file_path:
        return

    image = cv2.imread(
        file_path
    )

    if image is None:
        messagebox.showerror(
            "Lỗi",
            "Không thể đọc ảnh."
        )
        return

    original_image = image
    result_image = None
    image_path = file_path

    show_image(
        original_image,
        original_image_label,
        max_width=450,
        max_height=420
    )

    result_image_label.config(
        image="",
        text="Chưa có kết quả"
    )

    result_image_label.image = None

    psnr_value_label.config(
        text="PSNR: --"
    )

    ssim_value_label.config(
        text="SSIM: --"
    )

    image_name_label.config(
        text=Path(file_path).name
    )

    update_image_info()


# ============================================================
# KHÔI PHỤC ẢNH
# ============================================================

def restore_current_image():
    global result_image

    if original_image is None:
        messagebox.showwarning(
            "Thông báo",
            "Vui lòng mở ảnh trước."
        )
        return

    method = method_var.get()

    if not method:
        messagebox.showwarning(
            "Thông báo",
            "Vui lòng chọn phương pháp xử lý."
        )
        return

    kernel_size = get_kernel_size()

    try:
        result_image = restore_image(
            original_image,
            method,
            kernel_size
        )

        show_image(
            result_image,
            result_image_label,
            max_width=450,
            max_height=420
        )

        # Đánh giá ảnh
        metrics = evaluate_image(
            original_image,
            result_image
        )

        psnr = metrics["psnr"]
        ssim = metrics["ssim"]

        if np.isinf(psnr):
            psnr_text = "∞"
        else:
            psnr_text = f"{psnr:.2f} dB"

        psnr_value_label.config(
            text=f"PSNR: {psnr_text}"
        )

        ssim_value_label.config(
            text=f"SSIM: {ssim:.4f}"
        )

        status_var.set(
            f"Đã xử lý bằng {method}, Kernel = {kernel_size}"
        )

    except Exception as error:
        messagebox.showerror(
            "Lỗi xử lý",
            str(error)
        )


# ============================================================
# LƯU ẢNH
# ============================================================

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
            ("BMP", "*.bmp")
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

        messagebox.showinfo(
            "Thành công",
            f"Đã lưu ảnh:\n{save_path}"
        )

        status_var.set(
            "Đã lưu ảnh kết quả."
        )

    except Exception as error:
        messagebox.showerror(
            "Lỗi",
            str(error)
        )


# ============================================================
# SO SÁNH PHƯƠNG PHÁP
# ============================================================

def compare_current_image():
    if original_image is None:
        messagebox.showwarning(
            "Thông báo",
            "Vui lòng mở ảnh trước."
        )
        return

    kernel_size = get_kernel_size()

    try:
        results = compare_methods(
            original_image,
            kernel_size
        )

        if not results:
            return

        compare_window = tk.Toplevel(
            root
        )

        compare_window.title(
            "So sánh các phương pháp"
        )

        compare_window.geometry(
            "1250x720"
        )

        compare_window.minsize(
            1000,
            600
        )

        # ----------------------------------------------------
        # Tiêu đề
        # ----------------------------------------------------

        title_label = tk.Label(
            compare_window,
            text=(
                f"So sánh 5 phương pháp - "
                f"Kernel = {kernel_size}"
            ),
            font=("Arial", 16, "bold")
        )

        title_label.pack(
            pady=(12, 5)
        )

        description_label = tk.Label(
            compare_window,
            text=(
                "PSNR và SSIM được tính giữa "
                "ảnh đầu vào và ảnh sau xử lý."
            ),
            font=("Arial", 10)
        )

        description_label.pack(
            pady=(0, 10)
        )

        # ----------------------------------------------------
        # Khung chứa kết quả
        # ----------------------------------------------------

        outer_frame = tk.Frame(
            compare_window
        )

        outer_frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        comparison_container = tk.Frame(
            outer_frame
        )

        comparison_container.pack(
            expand=True
        )

        # ----------------------------------------------------
        # Hiển thị 5 phương pháp
        # ----------------------------------------------------

        for column, (
            method_name,
            image
        ) in enumerate(results.items()):

            method_frame = tk.Frame(
                comparison_container,
                relief="groove",
                borderwidth=1,
                padx=5,
                pady=5
            )

            method_frame.grid(
                row=0,
                column=column,
                padx=4,
                pady=5,
                sticky="n"
            )

            method_label = tk.Label(
                method_frame,
                text=method_name,
                font=("Arial", 11, "bold")
            )

            method_label.pack(
                pady=(2, 5)
            )

            image_label = tk.Label(
                method_frame,
                text="Đang hiển thị..."
            )

            image_label.pack(
                padx=3,
                pady=3
            )

            show_image(
                image,
                image_label,
                max_width=200,
                max_height=300
            )

            # ------------------------------------------------
            # PSNR / SSIM
            # ------------------------------------------------

            try:
                metrics = evaluate_image(
                    original_image,
                    image
                )

                psnr = metrics["psnr"]
                ssim = metrics["ssim"]

                if np.isinf(psnr):
                    psnr_text = "∞"
                else:
                    psnr_text = (
                        f"{psnr:.2f} dB"
                    )

                ssim_text = (
                    f"{ssim:.4f}"
                )

                psnr_label = tk.Label(
                    method_frame,
                    text=f"PSNR: {psnr_text}",
                    font=("Arial", 9)
                )

                psnr_label.pack(
                    pady=(5, 2)
                )

                ssim_label = tk.Label(
                    method_frame,
                    text=f"SSIM: {ssim_text}",
                    font=("Arial", 9)
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
                    pady=5
                )

        # ----------------------------------------------------
        # Nút đóng
        # ----------------------------------------------------

        close_button = ttk.Button(
            compare_window,
            text="Đóng",
            command=compare_window.destroy
        )

        close_button.pack(
            pady=(0, 12)
        )

    except Exception as error:
        messagebox.showerror(
            "Lỗi so sánh",
            str(error)
        )


# ============================================================
# BATCH PROCESSING
# ============================================================

def batch_processing():
    """
    Giao diện Batch Processing.

    Cho phép:
    - Chọn thư mục đầu vào.
    - Chọn phương pháp.
    - Điều chỉnh kernel.
    - Chọn thư mục đầu ra.
    - Xử lý nhiều ảnh.
    - Đánh giá PSNR / SSIM trung bình.
    - Hiển thị ảnh mẫu trước / sau.
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

    method = method_var.get()
    kernel_size = get_kernel_size()

    if not method:
        messagebox.showwarning(
            "Thông báo",
            "Vui lòng chọn phương pháp xử lý."
        )
        return

    try:
        # ----------------------------------------------------
        # Xử lý Batch
        # ----------------------------------------------------

        success, failed = process_batch(
            input_folder=input_folder,
            output_folder=output_folder,
            method=method,
            kernel_size=kernel_size
        )

        total = success + failed

        # ----------------------------------------------------
        # Tính PSNR / SSIM trung bình
        # ----------------------------------------------------

        psnr_values = []
        ssim_values = []

        supported_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        }

        input_path = Path(
            input_folder
        )

        output_path = Path(
            output_folder
        )

        sample_original = None
        sample_result = None

        for file_path in input_path.rglob("*"):

            if not file_path.is_file():
                continue

            if (
                file_path.suffix.lower()
                not in supported_extensions
            ):
                continue

            relative_path = (
                file_path.relative_to(
                    input_path
                )
            )

            output_file = (
                output_path / relative_path
            )

            if not output_file.exists():
                continue

            original = cv2.imread(
                str(file_path)
            )

            restored = cv2.imread(
                str(output_file)
            )

            if (
                original is None
                or restored is None
            ):
                continue

            try:
                metrics = evaluate_image(
                    original,
                    restored
                )

                if not np.isinf(
                    metrics["psnr"]
                ):
                    psnr_values.append(
                        metrics["psnr"]
                    )

                ssim_values.append(
                    metrics["ssim"]
                )

                # Lấy ảnh đầu tiên làm ảnh mẫu
                if sample_original is None:
                    sample_original = original
                    sample_result = restored

            except Exception:
                continue

        # ----------------------------------------------------
        # Tính trung bình
        # ----------------------------------------------------

        if psnr_values:
            avg_psnr = np.mean(
                psnr_values
            )
            avg_psnr_text = (
                f"{avg_psnr:.2f} dB"
            )
        else:
            avg_psnr_text = "--"

        if ssim_values:
            avg_ssim = np.mean(
                ssim_values
            )
            avg_ssim_text = (
                f"{avg_ssim:.4f}"
            )
        else:
            avg_ssim_text = "--"

        # ----------------------------------------------------
        # Cửa sổ kết quả
        # ----------------------------------------------------

        batch_window = tk.Toplevel(
            root
        )

        batch_window.title(
            "Kết quả Batch Processing"
        )

        batch_window.geometry(
            "1000x700"
        )

        batch_window.minsize(
            850,
            600
        )

        title = tk.Label(
            batch_window,
            text="KẾT QUẢ BATCH PROCESSING",
            font=("Arial", 16, "bold")
        )

        title.pack(
            pady=(15, 10)
        )

        info_frame = tk.Frame(
            batch_window
        )

        info_frame.pack(
            pady=5
        )

        tk.Label(
            info_frame,
            text=f"Phương pháp: {method}",
            font=("Arial", 11)
        ).grid(
            row=0,
            column=0,
            padx=20,
            pady=3
        )

        tk.Label(
            info_frame,
            text=f"Kernel: {kernel_size}",
            font=("Arial", 11)
        ).grid(
            row=0,
            column=1,
            padx=20,
            pady=3
        )

        tk.Label(
            info_frame,
            text=f"Tổng số ảnh: {total}",
            font=("Arial", 11)
        ).grid(
            row=1,
            column=0,
            padx=20,
            pady=3
        )

        tk.Label(
            info_frame,
            text=f"Thành công: {success}",
            font=("Arial", 11)
        ).grid(
            row=1,
            column=1,
            padx=20,
            pady=3
        )

        tk.Label(
            info_frame,
            text=f"Thất bại: {failed}",
            font=("Arial", 11)
        ).grid(
            row=2,
            column=0,
            padx=20,
            pady=3
        )

        tk.Label(
            info_frame,
            text=f"PSNR trung bình: {avg_psnr_text}",
            font=("Arial", 11, "bold")
        ).grid(
            row=2,
            column=1,
            padx=20,
            pady=3
        )

        tk.Label(
            info_frame,
            text=f"SSIM trung bình: {avg_ssim_text}",
            font=("Arial", 11, "bold")
        ).grid(
            row=3,
            column=0,
            columnspan=2,
            padx=20,
            pady=5
        )

        # ----------------------------------------------------
        # Ảnh mẫu trước / sau
        # ----------------------------------------------------

        sample_title = tk.Label(
            batch_window,
            text="Ảnh mẫu trước và sau xử lý",
            font=("Arial", 12, "bold")
        )

        sample_title.pack(
            pady=(15, 5)
        )

        sample_frame = tk.Frame(
            batch_window
        )

        sample_frame.pack(
            expand=True
        )

        before_frame = tk.Frame(
            sample_frame
        )

        before_frame.grid(
            row=0,
            column=0,
            padx=20
        )

        after_frame = tk.Frame(
            sample_frame
        )

        after_frame.grid(
            row=0,
            column=1,
            padx=20
        )

        tk.Label(
            before_frame,
            text="Ảnh trước",
            font=("Arial", 11, "bold")
        ).pack(
            pady=5
        )

        tk.Label(
            after_frame,
            text="Ảnh sau",
            font=("Arial", 11, "bold")
        ).pack(
            pady=5
        )

        before_label = tk.Label(
            before_frame,
            text="Không có ảnh"
        )

        before_label.pack()

        after_label = tk.Label(
            after_frame,
            text="Không có ảnh"
        )

        after_label.pack()

        if (
            sample_original is not None
            and sample_result is not None
        ):
            show_image(
                sample_original,
                before_label,
                max_width=350,
                max_height=280
            )

            show_image(
                sample_result,
                after_label,
                max_width=350,
                max_height=280
            )

        ttk.Button(
            batch_window,
            text="Đóng",
            command=batch_window.destroy
        ).pack(
            pady=15
        )

        status_var.set(
            f"Batch hoàn tất: {success}/{total} ảnh thành công."
        )

    except Exception as error:
        messagebox.showerror(
            "Lỗi Batch Processing",
            str(error)
        )


# ============================================================
# HIỂN THỊ THÔNG TIN
# ============================================================

def show_about():
    messagebox.showinfo(
        "Project 1",
        "Ứng dụng khôi phục ảnh cũ\n\n"
        "Môn: Xử lý ảnh\n\n"
        "Nhóm:\n"
        "Nguyễn Hoàng Huy\n"
        "Nguyễn Đăng Khoa\n"
        "Nguyễn Viết Anh Khôi"
    )


# ============================================================
# GIAO DIỆN CHÍNH
# ============================================================

root = tk.Tk()

root.title(
    "Ứng dụng khôi phục ảnh cũ - Project 1"
)

root.geometry(
    "1150x800"
)

root.minsize(
    1000,
    700
)


# ============================================================
# STYLE
# ============================================================

style = ttk.Style()

try:
    style.theme_use(
        "clam"
    )
except tk.TclError:
    pass


# ============================================================
# BIẾN GIAO DIỆN
# ============================================================

method_var = tk.StringVar(
    value="Median Blur"
)

kernel_var = tk.IntVar(
    value=5
)

status_var = tk.StringVar(
    value="Sẵn sàng."
)


# ============================================================
# MENU
# ============================================================

menubar = tk.Menu(
    root
)

file_menu = tk.Menu(
    menubar,
    tearoff=0
)

file_menu.add_command(
    label="Mở ảnh",
    command=open_image
)

file_menu.add_command(
    label="Lưu kết quả",
    command=save_result
)

file_menu.add_separator()

file_menu.add_command(
    label="Thoát",
    command=root.destroy
)

menubar.add_cascade(
    label="Tệp",
    menu=file_menu
)

tool_menu = tk.Menu(
    menubar,
    tearoff=0
)

tool_menu.add_command(
    label="Khôi phục ảnh",
    command=restore_current_image
)

tool_menu.add_command(
    label="So sánh phương pháp",
    command=compare_current_image
)

tool_menu.add_command(
    label="Batch Processing",
    command=batch_processing
)

menubar.add_cascade(
    label="Xử lý",
    menu=tool_menu
)

help_menu = tk.Menu(
    menubar,
    tearoff=0
)

help_menu.add_command(
    label="Thông tin Project",
    command=show_about
)

menubar.add_cascade(
    label="Trợ giúp",
    menu=help_menu
)

root.config(
    menu=menubar
)


# ============================================================
# TIÊU ĐỀ
# ============================================================

title_frame = tk.Frame(
    root
)

title_frame.pack(
    fill="x",
    pady=(15, 5)
)

title_label = tk.Label(
    title_frame,
    text="ỨNG DỤNG KHÔI PHỤC ẢNH CŨ",
    font=("Arial", 20, "bold")
)

title_label.pack()

subtitle_label = tk.Label(
    title_frame,
    text="Project 1 - Xử lý ảnh",
    font=("Arial", 11)
)

subtitle_label.pack(
    pady=(2, 5)
)


# ============================================================
# KHUNG ĐIỀU KHIỂN
# ============================================================

control_frame = ttk.LabelFrame(
    root,
    text="Điều khiển xử lý"
)

control_frame.pack(
    fill="x",
    padx=20,
    pady=10
)


# ------------------------------------------------------------
# Phương pháp
# ------------------------------------------------------------

ttk.Label(
    control_frame,
    text="Phương pháp:"
).grid(
    row=0,
    column=0,
    padx=(15, 5),
    pady=12
)

method_combo = ttk.Combobox(
    control_frame,
    textvariable=method_var,
    values=[
        "Median Blur",
        "Gaussian Blur",
        "Bilateral Filter",
        "Sharpening",
        "Inpainting"
    ],
    state="readonly",
    width=22
)

method_combo.grid(
    row=0,
    column=1,
    padx=5,
    pady=12
)


# ------------------------------------------------------------
# Kernel
# ------------------------------------------------------------

ttk.Label(
    control_frame,
    text="Kernel:"
).grid(
    row=0,
    column=2,
    padx=(20, 5),
    pady=12
)

kernel_spinbox = tk.Spinbox(
    control_frame,
    from_=3,
    to=31,
    increment=2,
    textvariable=kernel_var,
    width=8
)

kernel_spinbox.grid(
    row=0,
    column=3,
    padx=5,
    pady=12
)


# ------------------------------------------------------------
# Nút mở ảnh
# ------------------------------------------------------------

open_button = ttk.Button(
    control_frame,
    text="Mở ảnh",
    command=open_image
)

open_button.grid(
    row=0,
    column=4,
    padx=(25, 5),
    pady=12
)


# ------------------------------------------------------------
# Nút khôi phục
# ------------------------------------------------------------

restore_button = ttk.Button(
    control_frame,
    text="Khôi phục",
    command=restore_current_image
)

restore_button.grid(
    row=0,
    column=5,
    padx=5,
    pady=12
)


# ------------------------------------------------------------
# Nút so sánh
# ------------------------------------------------------------

compare_button = ttk.Button(
    control_frame,
    text="So sánh",
    command=compare_current_image
)

compare_button.grid(
    row=0,
    column=6,
    padx=5,
    pady=12
)


# ------------------------------------------------------------
# Nút Batch
# ------------------------------------------------------------

batch_button = ttk.Button(
    control_frame,
    text="Batch Processing",
    command=batch_processing
)

batch_button.grid(
    row=0,
    column=7,
    padx=5,
    pady=12
)


# ------------------------------------------------------------
# Nút lưu
# ------------------------------------------------------------

save_button = ttk.Button(
    control_frame,
    text="Lưu kết quả",
    command=save_result
)

save_button.grid(
    row=0,
    column=8,
    padx=(5, 15),
    pady=12
)


# ============================================================
# THÔNG TIN ẢNH
# ============================================================

info_frame = tk.Frame(
    root
)

info_frame.pack(
    fill="x",
    padx=25,
    pady=(0, 5)
)

image_name_label = tk.Label(
    info_frame,
    text="Chưa mở ảnh",
    font=("Arial", 10, "bold")
)

image_name_label.pack(
    side="left"
)

image_info_label = tk.Label(
    info_frame,
    text="Chưa mở ảnh",
    font=("Arial", 10)
)

image_info_label.pack(
    side="right"
)


# ============================================================
# KHUNG HIỂN THỊ ẢNH
# ============================================================

image_frame = tk.Frame(
    root
)

image_frame.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=10
)


# ------------------------------------------------------------
# Ảnh ban đầu
# ------------------------------------------------------------

original_frame = ttk.LabelFrame(
    image_frame,
    text="Ảnh ban đầu"
)

original_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 8)
)

original_image_label = tk.Label(
    original_frame,
    text="Chưa có ảnh",
    font=("Arial", 11)
)

original_image_label.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


# ------------------------------------------------------------
# Ảnh kết quả
# ------------------------------------------------------------

result_frame = ttk.LabelFrame(
    image_frame,
    text="Ảnh sau khi xử lý"
)

result_frame.pack(
    side="right",
    fill="both",
    expand=True,
    padx=(8, 0)
)

result_image_label = tk.Label(
    result_frame,
    text="Chưa có kết quả",
    font=("Arial", 11)
)

result_image_label.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


# ============================================================
# ĐÁNH GIÁ PSNR / SSIM
# ============================================================

evaluation_frame = ttk.LabelFrame(
    root,
    text="Đánh giá kết quả"
)

evaluation_frame.pack(
    fill="x",
    padx=20,
    pady=(0, 10)
)

psnr_value_label = tk.Label(
    evaluation_frame,
    text="PSNR: --",
    font=("Arial", 11, "bold")
)

psnr_value_label.pack(
    side="left",
    padx=50,
    pady=8
)

ssim_value_label = tk.Label(
    evaluation_frame,
    text="SSIM: --",
    font=("Arial", 11, "bold")
)

ssim_value_label.pack(
    side="left",
    padx=50,
    pady=8
)


# ============================================================
# STATUS BAR
# ============================================================

status_label = tk.Label(
    root,
    textvariable=status_var,
    anchor="w",
    relief="sunken",
    padx=10
)

status_label.pack(
    fill="x",
    side="bottom"
)


# ============================================================
# CHẠY GUI
# ============================================================

root.mainloop()