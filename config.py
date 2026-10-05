"""
config.py - Cấu hình chung cho project Hand Gesture Recognition
"""
import os

# ==================== Đường dẫn ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

# ==================== Danh sách cử chỉ ====================
GESTURES = [
    "fist",           # Nắm đấm
    "open_palm",      # Xòe bàn tay
    "thumbs_up",      # Giơ ngón cái lên
    "thumbs_down",    # Giơ ngón cái xuống
    "peace",          # Giơ 2 ngón (chữ V)
    "point",          # Chỉ tay (1 ngón trỏ)
    "swipe_left",     # Vuốt sang trái
    "swipe_right",    # Vuốt sang phải
    "pinch",          # Khép ngón tay (Click chuột)
]

NUM_CLASSES = len(GESTURES)

# ==================== Ảnh ====================
IMG_SIZE = 64           # Kích thước ảnh đầu vào (64x64)
IMG_CHANNELS = 3        # RGB

# ==================== Training Hyperparameters ====================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.8       # 80% train, 20% validation
EARLY_STOP_PATIENCE = 7 # Dừng sớm nếu val_loss không giảm sau N epochs

# ==================== Data Augmentation ====================
AUGMENTATION = {
    "random_horizontal_flip": 0.5,
    "random_rotation": 15,        # Xoay tối đa 15 độ
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
}

# ==================== Data Collection ====================
COLLECT_NUM_SAMPLES = 200   # Số ảnh cần thu thập cho mỗi cử chỉ
CAMERA_INDEX = 0            # Index camera (0 = webcam mặc định)

# ==================== Prediction ====================
CONFIDENCE_THRESHOLD = 0.6  # Ngưỡng confidence tối thiểu để thực thi action
COOLDOWN_SECONDS = 1.0      # Thời gian chờ giữa 2 lần thực thi action

# ==================== Swipe Detection ====================
SWIPE_DISTANCE_THRESHOLD = 80   # Pixel tối thiểu để coi là swipe
SWIPE_TIME_WINDOW = 0.5         # Giây - thời gian tối đa để hoàn thành swipe
