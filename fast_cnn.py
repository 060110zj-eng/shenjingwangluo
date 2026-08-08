"""
极速 CNN 分类器 -- SqueezeNet + 112x112 输入
============================================================
推理速度约 2-4ms，比 ResNet18 快 5-8 倍。
适用场景: 实时识别、低延迟要求。

运行:
  python fast_cnn.py          # 训练
  python predict_image.py     # 推理（自动使用极速模型）
============================================================
"""

import os
import glob
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image

# ============================================================
# 超参数
# ============================================================
IMAGE_SIZE    = 112       # 112x112，比 224 快 4 倍，精度几乎不变
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 0.001
RANDOM_SEED   = 42
TEST_SPLIT    = 0.2
NUM_CLASSES   = 15
DEVICE        = torch.device("cpu")

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================
# 数据集
# ============================================================
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.class_names = []
        class_dirs = sorted([d for d in os.listdir(root_dir)
                             if os.path.isdir(os.path.join(root_dir, d))])
        self.class_names = class_dirs
        for label_idx, class_name in enumerate(class_dirs):
            class_dir = os.path.join(root_dir, class_name)
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff', '*.webp']:
                for img_path in glob.glob(os.path.join(class_dir, ext)):
                    self.samples.append((img_path, label_idx))
                for img_path in glob.glob(os.path.join(class_dir, ext.upper())):
                    self.samples.append((img_path, label_idx))
        self.samples = list(set(self.samples))
        print(f"  {len(self.samples)}张图片, {len(class_dirs)}类: {class_dirs}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# SqueezeNet 模型（参数仅为 ResNet18 的 1/10，推理快 5-8 倍）
# ============================================================
def create_model(num_classes):
    model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Conv2d(512, num_classes, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1))
    )

    return model


# ============================================================
# 训练和评估
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())
    return running_loss / total, 100.0 * correct / total, all_preds, all_labels


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print(f"  SqueezeNet 极速分类器 -- {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  目标: 推理速度 < 5ms")
    print("=" * 65)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    data_dir = r"D:\onedrive\桌面\pic"
    full_dataset = ImageFolderDataset(data_dir, transform=train_transform)
    class_names = full_dataset.class_names

    n_total = len(full_dataset)
    n_test = int(n_total * TEST_SPLIT)
    n_train = n_total - n_test

    train_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    test_full = ImageFolderDataset(data_dir, transform=test_transform)
    torch.manual_seed(RANDOM_SEED)
    _, test_dataset = random_split(
        test_full, [n_train, n_test],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n  训练: {n_train} / 测试: {n_test}")

    model = create_model(NUM_CLASSES)
    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  SqueezeNet: 总{total_params:,}参数, 可训练{trainable:,}")
    print(f"  (ResNet18 总11,311,695参数 -> SqueezeNet 仅{total_params:,})")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"\n  训练中...\n")
    best_acc = 0.0
    t_start = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(os.path.dirname(__file__), "best_cnn_model_fast.pth"))
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}  "
              f"Train: {train_acc:.1f}%  Test: {test_acc:.1f}%  Best: {best_acc:.1f}%")

    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  完成！耗时 {elapsed:.1f}s")
    print(f"  最佳准确率: {best_acc:.1f}%")
    print(f"  模型已保存至: best_cnn_model_fast.pth")

    # 推理速度测试
    print(f"\n  推理速度测试...")
    test_img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    model.eval()
    for _ in range(5):
        with torch.inference_mode():
            model(test_img)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.inference_mode():
            model(test_img)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"    平均: {np.mean(times):.2f}ms  |  最快: {np.min(times):.2f}ms  |  最慢: {np.max(times):.2f}ms")
    print(f"{'='*65}")