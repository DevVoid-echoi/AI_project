"""
dataset.py - Dataset class và data augmentation cho training
"""
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from config import (
    DATASET_DIR, IMG_SIZE, GESTURES, TRAIN_SPLIT,
    BATCH_SIZE, AUGMENTATION
)


class HandGestureDataset(Dataset):
    """
    PyTorch Dataset để load ảnh cử chỉ tay từ thư mục.
    
    Cấu trúc thư mục:
        dataset/
        ├── fist/
        │   ├── 001.jpg
        │   ├── 002.jpg
        │   └── ...
        ├── open_palm/
        │   └── ...
        ├── swipe_left/
        │   └── ...
        └── swipe_right/
            └── ...
    """

    def __init__(self, root_dir=DATASET_DIR, transform=None):
        """
        Args:
            root_dir: Đường dẫn đến thư mục dataset
            transform: Torchvision transforms để áp dụng
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # List of (image_path, label_index)
        self.classes = GESTURES

        # Load tất cả ảnh từ các thư mục con
        for label_idx, gesture_name in enumerate(self.classes):
            gesture_dir = os.path.join(root_dir, gesture_name)
            if not os.path.exists(gesture_dir):
                print(f"⚠ Cảnh báo: Thư mục '{gesture_name}' không tồn tại, bỏ qua.")
                continue
            
            for img_name in os.listdir(gesture_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(gesture_dir, img_name)
                    self.samples.append((img_path, label_idx))

        print(f"✓ Loaded {len(self.samples)} ảnh từ {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load ảnh và chuyển sang RGB
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transform():
    """Transform cho training data (có augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=AUGMENTATION["random_horizontal_flip"]),
        transforms.RandomRotation(degrees=AUGMENTATION["random_rotation"]),
        transforms.ColorJitter(
            brightness=AUGMENTATION["color_jitter_brightness"],
            contrast=AUGMENTATION["color_jitter_contrast"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],    # ImageNet std
        ),
    ])


def get_val_transform():
    """Transform cho validation data (không augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_inference_transform():
    """Transform cho inference (giống val, dùng khi predict real-time)."""
    return get_val_transform()


def get_dataloaders(root_dir=DATASET_DIR, batch_size=BATCH_SIZE):
    """
    Tạo train và validation DataLoader.
    
    Returns:
        train_loader, val_loader, dataset
    """
    # Tạo full dataset với train transform
    full_dataset = HandGestureDataset(root_dir=root_dir, transform=get_train_transform())

    if len(full_dataset) == 0:
        raise ValueError(
            f"Dataset trống! Hãy thu thập dữ liệu trước bằng:\n"
            f"  python main.py --mode collect"
        )

    # Split train/val
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Ghi đè transform cho val_dataset (bỏ augmentation)
    # Lưu ý: random_split dùng subset nên ta cần wrapper
    val_dataset.dataset = HandGestureDataset(root_dir=root_dir, transform=get_val_transform())

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"✓ Train: {train_size} samples | Val: {val_size} samples")
    return train_loader, val_loader, full_dataset


# === Test nhanh ===
if __name__ == "__main__":
    print("Kiểm tra dataset...")
    try:
        train_loader, val_loader, dataset = get_dataloaders()
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}, Labels: {labels[:5]}")
            break
    except ValueError as e:
        print(f"Lỗi: {e}")
