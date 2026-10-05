"""
collect_data.py - Thu thập dữ liệu cử chỉ tay từ webcam
Dùng MediaPipe Tasks API (v1.0+) để detect tay, crop vùng tay, lưu ảnh vào thư mục theo label.
"""
import os
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import DATASET_DIR, GESTURES, CAMERA_INDEX, COLLECT_NUM_SAMPLES, IMG_SIZE, BASE_DIR


# Đường dẫn tới file model hand_landmarker
HAND_LANDMARKER_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")


def safe_imwrite(filepath, img):
    """
    Ghi ảnh ra file, hỗ trợ đường dẫn Unicode trên Windows.
    cv2.imwrite() bị lỗi âm thầm với đường dẫn chứa ký tự đặc biệt (tiếng Việt),
    nên dùng cv2.imencode() + ghi bytes bằng Python thay thế.
    """
    success, encoded = cv2.imencode('.jpg', img)
    if success:
        with open(filepath, 'wb') as f:
            f.write(encoded.tobytes())
        return True
    return False


def create_dataset_dirs():
    """Tạo thư mục cho từng gesture."""
    for gesture in GESTURES:
        gesture_dir = os.path.join(DATASET_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
    print(f"✓ Đã tạo {len(GESTURES)} thư mục trong {DATASET_DIR}")


def get_hand_region(frame, hand_landmarks, padding=40):
    """
    Cắt vùng chứa bàn tay từ frame dựa trên landmarks.
    
    Args:
        frame: Ảnh gốc từ webcam
        hand_landmarks: MediaPipe NormalizedLandmark list
        padding: Pixel mở rộng xung quanh bounding box
        
    Returns:
        Ảnh vùng tay đã crop, hoặc None nếu không valid
    """
    h, w, _ = frame.shape
    
    # Lấy bounding box từ landmarks
    x_coords = [lm.x * w for lm in hand_landmarks]
    y_coords = [lm.y * h for lm in hand_landmarks]
    
    x_min = int(max(0, min(x_coords) - padding))
    y_min = int(max(0, min(y_coords) - padding))
    x_max = int(min(w, max(x_coords) + padding))
    y_max = int(min(h, max(y_coords) + padding))
    
    if x_max - x_min < 10 or y_max - y_min < 10:
        return None
    
    hand_crop = frame[y_min:y_max, x_min:x_max]
    hand_crop = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
    return hand_crop


def collect_gesture_data():
    """
    Chạy vòng lặp thu thập dữ liệu.
    
    Điều khiển:
        - Nhấn 's' để bắt đầu/tạm dừng chụp tự động
        - Nhấn 'c' để chụp 1 ảnh
        - Nhấn 'n' để chuyển sang gesture tiếp theo
        - Nhấn 'q' để thoát
    """
    create_dataset_dirs()
    
    # Khởi tạo MediaPipe HandLandmarker (Tasks API v1.0+)
    # Đọc model dưới dạng bytes để tránh lỗi Unicode path trên Windows
    with open(HAND_LANDMARKER_PATH, "rb") as f:
        model_data = f.read()
    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("✗ Không mở được webcam!")
        return

    current_gesture_idx = 0
    count = 0
    auto_capture = False
    last_capture_time = 0
    capture_interval = 0.15  # Chụp tự động mỗi 0.15s

    print("\n" + "=" * 60)
    print("  THU THẬP DỮ LIỆU CỬ CHỈ TAY")
    print("=" * 60)
    print("  [S] Bật/Tắt chụp tự động")
    print("  [C] Chụp 1 ảnh")
    print("  [N] Gesture tiếp theo")
    print("  [Q] Thoát")
    print("=" * 60)

    while current_gesture_idx < len(GESTURES):
        gesture = GESTURES[current_gesture_idx]
        gesture_dir = os.path.join(DATASET_DIR, gesture)
        
        # Đếm ảnh đã có
        existing = len([f for f in os.listdir(gesture_dir) 
                       if f.endswith(('.jpg', '.png'))])
        count = existing

        print(f"\n→ Thu thập: [{gesture}] ({count}/{COLLECT_NUM_SAMPLES})")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror

            # Chuyển sang RGB và tạo MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect hand
            detection_result = hand_landmarker.detect(mp_image)

            hand_crop = None

            if detection_result.hand_landmarks:
                for landmarks in detection_result.hand_landmarks:
                    # Vẽ landmarks lên frame thật (để hiển thị cho người dùng xem)
                    _draw_landmarks_on_frame(frame, landmarks)
                    
                    # TẠO ẢNH KHUNG XƯƠNG TRÊN NỀN ĐEN (để AI học)
                    black_canvas = np.zeros_like(frame)
                    _draw_landmarks_on_frame(black_canvas, landmarks)
                    
                    # Cắt vùng tay từ ảnh khung xương thay vì ảnh thật
                    hand_crop = get_hand_region(black_canvas, landmarks)

            # Hiển thị thông tin
            status_text = f"Gesture: {gesture} | Count: {count}/{COLLECT_NUM_SAMPLES}"
            mode_text = "AUTO CAPTURE" if auto_capture else "MANUAL"
            
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, mode_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 0, 255) if auto_capture else (200, 200, 200), 2)

            if hand_crop is not None:
                # Hiển thị preview crop
                preview = cv2.resize(hand_crop, (128, 128))
                frame[10:138, frame.shape[1]-138:frame.shape[1]-10] = preview

            # Auto capture
            if auto_capture and hand_crop is not None:
                current_time = time.time()
                if current_time - last_capture_time >= capture_interval:
                    filename = f"{gesture}_{count:04d}.jpg"
                    filepath = os.path.join(gesture_dir, filename)
                    safe_imwrite(filepath, hand_crop)
                    count += 1
                    last_capture_time = current_time
                    
                    if count >= COLLECT_NUM_SAMPLES:
                        print(f"  ✓ Đã đủ {COLLECT_NUM_SAMPLES} ảnh cho [{gesture}]!")
                        auto_capture = False
                        break

            cv2.imshow("Collect Data - Hand Gesture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                hand_landmarker.close()
                print("\n✓ Thoát thu thập dữ liệu.")
                return
            elif key == ord('s'):
                auto_capture = not auto_capture
                print(f"  {'▶ Bắt đầu' if auto_capture else '⏸ Tạm dừng'} chụp tự động")
            elif key == ord('c') and hand_crop is not None:
                filename = f"{gesture}_{count:04d}.jpg"
                filepath = os.path.join(gesture_dir, filename)
                safe_imwrite(filepath, hand_crop)
                count += 1
                print(f"  📷 Đã chụp: {filename} ({count}/{COLLECT_NUM_SAMPLES})")
            elif key == ord('n'):
                break

        current_gesture_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    print("\n✓ Hoàn tất thu thập dữ liệu!")


def _draw_landmarks_on_frame(frame, landmarks):
    """
    Vẽ khung xương tay lên frame với màu sắc phân biệt từng ngón tay.
    Mỗi ngón tay có một màu riêng để CNN dễ học đặc trưng hình dạng hơn.
    Nét vẽ dày và điểm to để không bị mất chi tiết khi resize xuống 64x64.
    """
    h, w, _ = frame.shape

    # Màu sắc riêng cho từng ngón tay (BGR format)
    FINGER_COLORS = {
        "thumb":  (0, 128, 255),   # Cam - Ngón cái
        "index":  (0, 255, 0),     # Xanh lá - Ngón trỏ
        "middle": (255, 255, 0),   # Vàng - Ngón giữa
        "ring":   (255, 0, 255),   # Tím - Ngón áp út
        "pinky":  (0, 255, 255),   # Xanh cyan - Ngón út
        "palm":   (200, 200, 200), # Xám - Lòng bàn tay
    }

    # Kết nối cho từng ngón tay
    FINGER_CONNECTIONS = {
        "thumb":  [(0, 1), (1, 2), (2, 3), (3, 4)],
        "index":  [(0, 5), (5, 6), (6, 7), (7, 8)],
        "middle": [(0, 9), (9, 10), (10, 11), (11, 12)],
        "ring":   [(0, 13), (13, 14), (14, 15), (15, 16)],
        "pinky":  [(0, 17), (17, 18), (18, 19), (19, 20)],
        "palm":   [(5, 9), (9, 13), (13, 17), (0, 5), (0, 17)],
    }

    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    LINE_THICKNESS = 4   # Nét dày để rõ sau khi resize
    JOINT_RADIUS   = 6   # Điểm khớp to

    # Vẽ từng ngón tay với màu riêng
    for finger_name, connections in FINGER_CONNECTIONS.items():
        color = FINGER_COLORS[finger_name]
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx],
                         color, LINE_THICKNESS, cv2.LINE_AA)

    # Vẽ các điểm khớp
    for i, point in enumerate(points):
        # Ngón cái (0-4)
        if i <= 4:   color = FINGER_COLORS["thumb"]
        elif i <= 8:  color = FINGER_COLORS["index"]
        elif i <= 12: color = FINGER_COLORS["middle"]
        elif i <= 16: color = FINGER_COLORS["ring"]
        else:         color = FINGER_COLORS["pinky"]
        cv2.circle(frame, point, JOINT_RADIUS, color, -1, cv2.LINE_AA)
        # Viền trắng xung quanh điểm khớp để nổi bật hơn
        cv2.circle(frame, point, JOINT_RADIUS + 1, (255, 255, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    collect_gesture_data()

