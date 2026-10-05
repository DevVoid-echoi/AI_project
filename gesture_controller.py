"""
gesture_controller.py - Ánh xạ cử chỉ tay sang hành động điều khiển máy tính
"""
import time
import pyautogui
from config import COOLDOWN_SECONDS

# Tắt failsafe pyautogui (cần thiết nếu chuột di chuyển nhiều)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # Chỉnh về 0 để loại bỏ hoàn toàn độ trễ 50ms mặc định


class GestureController:
    """
    Điều khiển máy tính dựa trên cử chỉ tay.
    
    Bảng ánh xạ:
        fist        → Play/Pause media
        open_palm   → Mute/Unmute
        thumbs_up   → Tăng âm lượng
        thumbs_down → Giảm âm lượng
        peace       → Chụp screenshot
        point       → (Dành cho di chuột - cần xử lý riêng)
        swipe_left  → Slide trước (phím Left)
        swipe_right → Slide tiếp (phím Right)
    """

    def __init__(self):
        self._last_action_time = {}  # gesture → last_executed_time
        self.last_gesture = None     # Track last gesture for edge-triggering
        self._volume_controller = None
        self._init_volume_control()
        
        # Biến phục vụ làm mượt chuột (Exponential Moving Average)
        self.prev_x = 0
        self.prev_y = 0
        self.smoothing_factor = 0.6  # Tăng lên 0.6 để chuột phản hồi nhanh hơn, bám sát tay hơn

    def _init_volume_control(self):
        """Khởi tạo pycaw cho điều khiển âm lượng Windows."""
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None
            )
            self._volume_controller = cast(
                interface, POINTER(IAudioEndpointVolume)
            )
            print("✓ Volume controller initialized")
        except Exception as e:
            print(f"⚠ Không thể khởi tạo volume controller: {e}")
            print("  Sẽ dùng phím tắt thay thế.")
            self._volume_controller = None

    def _is_cooldown(self, gesture):
        """Kiểm tra xem gesture có đang trong thời gian cooldown không."""
        # Cử chỉ di chuyển chuột ('point') cần cập nhật liên tục (60 FPS), KHÔNG dùng cooldown
        if gesture == "point":
            return False
            
        now = time.time()
        last_time = self._last_action_time.get(gesture, 0)
        
        # Âm lượng cần kích hoạt liên tục nhưng nhanh hơn (0.3s thay vì 1s)
        cooldown_limit = 0.3 if gesture in ["thumbs_up", "thumbs_down"] else COOLDOWN_SECONDS

        if now - last_time < cooldown_limit:
            return True
        self._last_action_time[gesture] = now
        return False

    def execute(self, gesture, extra_data=None):
        """
        Thực thi action tương ứng với gesture.
        
        Args:
            gesture: Tên cử chỉ (string)
            extra_data: Dữ liệu bổ sung (vd: tọa độ ngón tay cho 'point')
            
        Returns:
            Mô tả action đã thực hiện, hoặc None nếu cooldown
        """
        # Nếu mất dấu tay hoặc không có cử chỉ -> reset trạng thái
        if not gesture:
            self.last_gesture = None
            return None

        # Danh sách các cử chỉ chỉ kích hoạt 1 lần duy nhất cho đến khi hạ tay xuống
        one_shot_gestures = ["fist", "open_palm", "peace", "pinch", "swipe_left", "swipe_right"]

        # Nếu cử chỉ này thuộc loại 1-lần và GIỐNG HỆT cử chỉ của frame trước -> Bỏ qua
        if gesture in one_shot_gestures and gesture == self.last_gesture:
            return None

        # Cập nhật trạng thái cử chỉ hiện tại
        self.last_gesture = gesture

        # Kiểm tra cooldown (tránh rung nhiễu/debounce)
        if self._is_cooldown(gesture):
            return None

        action_map = {
            "fist": self._play_pause,
            "open_palm": self._mute_unmute,
            "thumbs_up": self._volume_up,
            "thumbs_down": self._volume_down,
            "peace": self._screenshot,
            "point": self._move_cursor,
            "swipe_left": self._slide_previous,
            "swipe_right": self._slide_next,
            "pinch": self._left_click,
        }

        action_func = action_map.get(gesture)
        if action_func:
            return action_func(extra_data)
        return None

    # ==================== Actions ====================

    def _play_pause(self, _=None):
        """Play/Pause media."""
        pyautogui.press("playpause")
        return "▶⏸ Play/Pause"

    def _mute_unmute(self, _=None):
        """Mute/Unmute."""
        if self._volume_controller:
            muted = self._volume_controller.GetMute()
            self._volume_controller.SetMute(not muted, None)
            return "🔇 Mute" if not muted else "🔊 Unmute"
        else:
            pyautogui.press("volumemute")
            return "🔇 Toggle Mute"

    def _volume_up(self, _=None):
        """Tăng âm lượng 5%."""
        if self._volume_controller:
            current = self._volume_controller.GetMasterVolumeLevelScalar()
            new_vol = min(1.0, current + 0.05)
            self._volume_controller.SetMasterVolumeLevelScalar(new_vol, None)
            return f"🔊 Volume: {new_vol:.0%}"
        else:
            pyautogui.press("volumeup")
            return "🔊 Volume Up"

    def _volume_down(self, _=None):
        """Giảm âm lượng 5%."""
        if self._volume_controller:
            current = self._volume_controller.GetMasterVolumeLevelScalar()
            new_vol = max(0.0, current - 0.05)
            self._volume_controller.SetMasterVolumeLevelScalar(new_vol, None)
            return f"🔉 Volume: {new_vol:.0%}"
        else:
            pyautogui.press("volumedown")
            return "🔉 Volume Down"

    def _screenshot(self, _=None):
        """Chụp screenshot."""
        pyautogui.hotkey("win", "shift", "s")  # Windows Snipping Tool
        return "📸 Screenshot"

    def _move_cursor(self, extra_data=None):
        """Di chuyển chuột theo vị trí ngón trỏ (Có làm mượt bằng EMA)."""
        if extra_data and "x" in extra_data and "y" in extra_data:
            screen_w, screen_h = pyautogui.size()
            # Định nghĩa "Vùng hoạt động" (Active Region) ở giữa camera.
            # Bỏ đi 20% lề xung quanh. Nếu tay nằm trong 60% ở giữa này, nó sẽ bao phủ toàn bộ màn hình.
            margin_x = 0.2
            margin_y = 0.2
            active_w = 1.0 - 2 * margin_x
            active_h = 1.0 - 2 * margin_y

            # Tính toán vị trí tương đối của tay trong vùng hoạt động
            norm_x = (extra_data["x"] - margin_x) / active_w
            norm_y = (extra_data["y"] - margin_y) / active_h

            # Giới hạn giá trị trong khoảng [0, 1] để chuột không văng ra ngoài mép
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            
            # Tọa độ thô đã được khuếch đại
            raw_x = norm_x * screen_w
            raw_y = norm_y * screen_h
            
            # Khởi tạo giá trị nếu là lần đầu di chuyển
            if self.prev_x == 0 and self.prev_y == 0:
                self.prev_x, self.prev_y = raw_x, raw_y

            # Áp dụng thuật toán Exponential Moving Average (EMA)
            smooth_x = self.prev_x + self.smoothing_factor * (raw_x - self.prev_x)
            smooth_y = self.prev_y + self.smoothing_factor * (raw_y - self.prev_y)

            # Cập nhật lịch sử
            self.prev_x, self.prev_y = smooth_x, smooth_y

            # Di chuyển chuột thực tế (duration=0 để tránh delay của pyautogui)
            pyautogui.moveTo(int(smooth_x), int(smooth_y), duration=0)
            return f"🖱 Cursor → ({int(smooth_x)}, {int(smooth_y)})"
        return None

    def _slide_previous(self, _=None):
        """Chuyển về slide trước (phím Left Arrow)."""
        pyautogui.press("left")
        return "⬅ Slide trước"

    def _slide_next(self, _=None):
        """Chuyển sang slide tiếp (phím Right Arrow)."""
        pyautogui.press("right")
        return "➡ Slide tiếp"

    def _left_click(self, _=None):
        """Click chuột trái."""
        pyautogui.click()
        return "🖱 Left Click"


if __name__ == "__main__":
    print("Test GestureController:")
    controller = GestureController()
    
    # Test từng gesture
    for gesture in ["fist", "thumbs_up", "thumbs_down", "swipe_left", "swipe_right"]:
        result = controller.execute(gesture)
        print(f"  {gesture}: {result}")
        time.sleep(0.2)
