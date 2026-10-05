"""
train.py - Pipeline huấn luyện CNN model
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from dataset import get_dataloaders
from config import (
    MODEL_SAVE_PATH, NUM_CLASSES, NUM_EPOCHS,
    LEARNING_RATE, EARLY_STOP_PATIENCE
)


def train_model():
    """Huấn luyện model CNN với early stopping."""
    
    # === Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Device: {device}")
    
    # Load data
    train_loader, val_loader, dataset = get_dataloaders()
    
    # Tạo model
    model = get_model(num_classes=NUM_CLASSES, device=device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Tạo thư mục lưu model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # === Training Loop ===
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"  TRAINING CNN - {NUM_EPOCHS} epochs")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total * 100

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total if val_total > 0 else 1
        val_acc = val_correct / val_total * 100 if val_total > 0 else 0

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%"
        )

        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  💾 Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n⏹  Early stopping tại epoch {epoch+1}")
                break

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training hoàn tất trong {elapsed:.1f}s")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Model saved: {MODEL_SAVE_PATH}")
    print(f"{'='*60}")

    return history


if __name__ == "__main__":
    train_model()
