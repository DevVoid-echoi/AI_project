"""
model.py - Định nghĩa kiến trúc CNN cho nhận diện cử chỉ tay
"""
import torch
import torch.nn as nn
from config import IMG_SIZE, IMG_CHANNELS, NUM_CLASSES


class HandGestureCNN(nn.Module):
    """
    CNN 3 lớp convolution cho bài toán phân loại cử chỉ tay.
    
    Kiến trúc:
        Conv Block 1: Conv2d(3, 32) → BatchNorm → ReLU → MaxPool
        Conv Block 2: Conv2d(32, 64) → BatchNorm → ReLU → MaxPool
        Conv Block 3: Conv2d(64, 128) → BatchNorm → ReLU → MaxPool
        Classifier:   Flatten → FC(128*8*8, 512) → ReLU → Dropout
                       → FC(512, 128) → ReLU → Dropout → FC(128, num_classes)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(HandGestureCNN, self).__init__()

        # === Convolutional Feature Extractor ===
        self.features = nn.Sequential(
            # Block 1: 64x64x3 → 32x32x32
            nn.Conv2d(IMG_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32x32x32 → 16x16x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 16x16x64 → 8x8x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Tính kích thước feature map sau conv layers
        self._feature_size = 128 * (IMG_SIZE // 8) * (IMG_SIZE // 8)  # 128 * 8 * 8

        # === Classifier ===
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor shape (batch, 3, 64, 64)
        Returns:
            logits: Tensor shape (batch, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(num_classes=NUM_CLASSES, device="cpu"):
    """Tạo model và chuyển sang device."""
    model = HandGestureCNN(num_classes=num_classes)
    model = model.to(device)
    return model


def load_model(model_path, num_classes=NUM_CLASSES, device="cpu"):
    """Load model đã train từ file .pth."""
    model = get_model(num_classes=num_classes, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# === Test nhanh model ===
if __name__ == "__main__":
    model = get_model()
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    output = model(dummy_input)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Num classes:  {NUM_CLASSES}")
    
    # Đếm parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
