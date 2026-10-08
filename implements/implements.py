import numpy as np

def conv(x, w, b, stride=1, padding=1):
    """
    c: Số kênh màu (R,G,B = 3)
    h = w: Chiều cao và chiều rộng (h = w = 64)
    stride: Mỗi lần trượt bao nhiêu ô
    k: Kích thước kernel
    padding: Thêm bao nhiêu viền 0 để giữ nguyên kích thước

    x: (C_in, H, W)         - Ảnh vào
    w: (C_out, C_in, K, K)  - C_out bộ lọc
    b: (C_out,)             - Bias cho mỗi bộ lọc
    """
    c_in, H, W = x.shape
    c_out, _, k, _ = w.shape

    # Thêm padding
    x_pad = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))

    # Tính kích thước đầu ra: n_out = (n_in + 2*padding - k) // stride + 1
    H_out = (H + 2 * padding - k) // stride + 1
    W_out = (W + 2*padding - k) // stride + 1
    out = np.zeros((c_out, H_out, W_out))

    # Trượt kernel
    for co in range(c_out): # Với từng bộ lọc
        for i in range (H_out): # Với từng hàng
            for j in range (W_out): # Với từng cột
                r, c = i * stride, j * stride
                region = x_pad[:, r: r + k, c: c + k]
                out[co, i, j] = np.sum(region * w[co]) + b[co]

    return out

def relu(x):
    """
    Hàm kích hoạt, giá trị nhỏ nhất của từng số trong ma trận là 0
    => Tạo nên tính phi tuyến cho phép chồng các lớp tuyến tính lên nhau
    """
    return np.maximum(0, x)

def maxpool(x, k=2):
    """
    Chia bản đồ thành các ô 2x2 và giữ giá trị lớn nhất của mỗi ô
    => Kích thước giảm 1 nửa, lượng tính toán giảm, mạng chịu được sự xê dịch
    """
    c, h, w = x.shape
    x = x[:, :h // k * k, :w // k * k]
    return x.reshape(c, h // k, k, w // k, k).max(axis=(2,4))

def flatten(x):
    """
    Trải thẳng sơ đồ 2d thành vector 1d
    """
    return x.reshape(-1)

def linear(x, W, b):
    """
    Fully connected
    Nhân với ma trận trọng số và cộng bias
    (Ma trận trọng số W sẽ cập nhật sau mỗi lần học bằng backpropagation)
    """
    return W @ x + b

def softmax(z): 
    """
    Tính xác suất có tổng = 1
    z: Điểm số sau khi fully connected 
    e^z_i / sum(e^z-j) j=0->k
    """
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()

def cross_entropy(probs, y):
    """
    Tính loss, nếu gần đúng thì cross_entropy trả lại nhỏ, nếu sai thì cross_entropy trả lại lớn
    -log(p_y)
    """
    return -np.log(probs[y] + 1e-12)

def forward_with_cache(x, params):
    """
    Lan truyền xuôi qua toàn mạng, đồng thời lưu lại (cache) các giá trị trung gian
    => Backward cần các giá trị này để tính gradient

    Kiến trúc (ví dụ ảnh 3x64x64, kênh 32/64/128):
        [Conv → ReLU → MaxPool] x3 : 3x64x64 → 32x32x32 → 64x16x16 → 128x8x8
        Flatten                    : 128x8x8 → 8192
        Linear → ReLU              : 8192 → 512
        Linear                     : 512 → 9 (logits)

    x: (C_in, H, W)  - Ảnh vào
    params: dict chứa trọng số w1..w5 và bias b1..b5

    Trả về:
        logits: (num_classes,) - Điểm số thô của từng lớp (chưa qua softmax)
        cache: dict các giá trị trung gian
    """
    cache = {}

    # 3 khối tích chập: Conv → ReLU → MaxPool
    for i in (1, 2, 3):
        cache[f"x{i}"] = x                              # Input của conv i  → conv_backward cần để tính dw
        z = conv(x, params[f"w{i}"], params[f"b{i}"])
        cache[f"z{i}"] = z                              # Input của ReLU    → relu_backward cần biết chỗ nào > 0
        a = relu(z)
        cache[f"a{i}"] = a                              # Input của MaxPool → maxpool_backward cần biết vị trí max
        x = maxpool(a)                                  # Kích thước giảm 1 nửa

    # Duỗi feature map thành vector
    cache["pool_shape"] = x.shape                       # flatten_backward cần để reshape ngược lại
    f = flatten(x)
    cache["f"] = f                                      # Input của linear 4 → tính dW4

    # Lớp fully connected ẩn
    z4 = linear(f, params["w4"], params["b4"])
    cache["z4"] = z4                                    # Input của ReLU     → relu_backward
    a4 = relu(z4)
    cache["a4"] = a4                                    # Input của linear 5 → tính dW5

    # Lớp đầu ra: điểm số cho từng lớp (softmax tính riêng ở ngoài)
    logits = linear(a4, params["w5"], params["b5"])
    return logits, cache