import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import sys
import os

class RedirectText:
    def __init__(self, text_ctrl):
        self.output = text_ctrl

    def write(self, string):
        self.output.insert(tk.END, string)
        self.output.see(tk.END)

    def flush(self):
        pass

class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Controller AI")
        self.root.geometry("750x650")
        self.root.configure(padx=20, pady=20)
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Segoe UI', 11), padding=10)
        style.configure('Small.TButton', font=('Segoe UI', 9), padding=5)
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'))
        
        # Title
        title_label = ttk.Label(root, text="🖐 Điều Khiển Máy Tính Bằng Cử Chỉ Tay", style='Title.TLabel')
        title_label.pack(pady=(0, 10))
        
        # === Hướng dẫn nhanh ===
        guide_frame = ttk.LabelFrame(root, text="📖 Hướng dẫn nhanh", padding=10)
        guide_frame.pack(fill=tk.X, pady=(0, 10))
        
        guide_text = (
            "Bước 1: Nhấn 'Thu thập dữ liệu' → Cửa sổ camera mở ra → "
            "Giơ tay trước camera → Nhấn phím S để bật chụp tự động → "
            "Nhấn N để chuyển sang cử chỉ tiếp → Nhấn Q khi xong.\n"
            "Bước 2: Nhấn 'Kiểm tra Dataset' để xem đã có đủ ảnh chưa.\n"
            "Bước 3: Nhấn 'Huấn luyện Model' → Đợi đến khi hoàn tất.\n"
            "Bước 4: Nhấn 'Chạy Điều Khiển' → Giơ tay để điều khiển máy tính!"
        )
        ttk.Label(guide_frame, text=guide_text, wraplength=680, font=('Segoe UI', 9)).pack()
        
        # Frame for main buttons
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill=tk.X, pady=5)
        
        # Main buttons
        self.btn_collect = ttk.Button(btn_frame, text="📸 1. Thu thập dữ liệu", command=self.run_collect)
        self.btn_collect.pack(side=tk.LEFT, expand=True, padx=3)
        
        self.btn_train = ttk.Button(btn_frame, text="🧠 2. Huấn luyện Model", command=self.run_train)
        self.btn_train.pack(side=tk.LEFT, expand=True, padx=3)
        
        self.btn_run = ttk.Button(btn_frame, text="🚀 3. Chạy Điều Khiển", command=self.run_controller)
        self.btn_run.pack(side=tk.LEFT, expand=True, padx=3)

        # Frame for utility buttons
        util_frame = ttk.Frame(root)
        util_frame.pack(fill=tk.X, pady=(5, 0))

        self.btn_check = ttk.Button(util_frame, text="📊 Kiểm tra Dataset", command=self.check_dataset, style='Small.TButton')
        self.btn_check.pack(side=tk.LEFT, padx=3)

        self.btn_clear = ttk.Button(util_frame, text="🗑 Xóa Console", command=self.clear_console, style='Small.TButton')
        self.btn_clear.pack(side=tk.LEFT, padx=3)

        # Console output area
        lbl_console = ttk.Label(root, text="Logs & Output:", font=('Segoe UI', 10, 'bold'))
        lbl_console.pack(anchor=tk.W, pady=(10, 5))
        
        self.console = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=15, font=('Consolas', 10), bg="#1e1e1e", fg="#00ff00")
        self.console.pack(fill=tk.BOTH, expand=True)
        
        # Redirect stdout and stderr
        redir = RedirectText(self.console)
        sys.stdout = redir
        sys.stderr = redir
        
        print("=" * 60)
        print("  Chào mừng! Hệ thống đã sẵn sàng.")
        print("=" * 60)
        print("")
        print("Khi cửa sổ camera mở ra (Bước 1):")
        print("  [S] = Bật/Tắt chụp tự động (quan trọng nhất!)")
        print("  [C] = Chụp 1 ảnh thủ công")
        print("  [N] = Chuyển sang cử chỉ tiếp theo")
        print("  [Q] = Thoát")
        print("")
        
    def _run_script_in_thread(self, target_func, script_name):
        """Chạy hàm trong thread để không block GUI."""
        def task():
            print(f"[{script_name}] Bắt đầu thực thi...")
            self.btn_collect.config(state=tk.DISABLED)
            self.btn_train.config(state=tk.DISABLED)
            self.btn_run.config(state=tk.DISABLED)
            
            try:
                target_func()
            except Exception as e:
                print(f"[LỖI] {e}")
            finally:
                print(f"[{script_name}] Đã hoàn tất hoặc bị đóng.\n")
                self.btn_collect.config(state=tk.NORMAL)
                self.btn_train.config(state=tk.NORMAL)
                self.btn_run.config(state=tk.NORMAL)

        threading.Thread(target=task, daemon=True).start()

    def check_dataset(self):
        """Kiểm tra và in ra số lượng ảnh đã thu thập cho mỗi cử chỉ."""
        from config import DATASET_DIR, GESTURES, COLLECT_NUM_SAMPLES
        
        print("=" * 50)
        print("  KIỂM TRA DATASET")
        print("=" * 50)
        
        total = 0
        all_ready = True
        
        for gesture in GESTURES:
            gesture_dir = os.path.join(DATASET_DIR, gesture)
            if os.path.exists(gesture_dir):
                count = len([f for f in os.listdir(gesture_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            else:
                count = 0
            
            total += count
            status = "✓" if count >= COLLECT_NUM_SAMPLES else "✗"
            if count < COLLECT_NUM_SAMPLES:
                all_ready = False
            
            bar_len = min(count, COLLECT_NUM_SAMPLES) * 20 // max(COLLECT_NUM_SAMPLES, 1)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {status} {gesture:15s} [{bar}] {count:4d}/{COLLECT_NUM_SAMPLES}")
        
        print(f"\n  Tổng cộng: {total} ảnh")
        
        if total == 0:
            print("\n  ⚠ CHƯA CÓ ẢNH NÀO!")
            print("  → Hãy nhấn '📸 1. Thu thập dữ liệu'")
            print("  → Khi camera mở, nhấn phím S để bắt đầu chụp tự động")
        elif all_ready:
            print("\n  ✓ Dataset đã sẵn sàng! Có thể chuyển sang Huấn luyện.")
        else:
            print("\n  ⚠ Một số cử chỉ chưa đủ ảnh.")
            print("  → Nhấn '📸 Thu thập dữ liệu' để chụp thêm.")
        
        print("=" * 50 + "\n")

    def clear_console(self):
        """Xóa nội dung console."""
        self.console.delete(1.0, tk.END)

    def run_collect(self):
        print("─" * 50)
        print("  HƯỚNG DẪN THU THẬP DỮ LIỆU:")
        print("  1. Cửa sổ camera sẽ mở ra")
        print("  2. Giơ tay trước camera")
        print("  3. Nhấn phím S để BẮT ĐẦU chụp tự động")
        print("  4. Thay đổi góc tay liên tục khi đang chụp")
        print("  5. Nhấn N khi muốn chuyển sang cử chỉ tiếp")
        print("  6. Nhấn Q khi hoàn tất tất cả")
        print("─" * 50)
        from collect_data import collect_gesture_data
        self._run_script_in_thread(collect_gesture_data, "Collect Data")

    def run_train(self):
        from train import train_model
        self._run_script_in_thread(train_model, "Train Model")

    def run_controller(self):
        from predict import run_realtime_controller
        self._run_script_in_thread(run_realtime_controller, "Run Controller")

if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureApp(root)
    root.mainloop()
