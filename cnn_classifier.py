"""
ResNet18 迁移学习 CNN 图片分类器
============================================================
策略: 使用 ImageNet 预训练权重，冻结底层卷积层，只训练顶层分类器。
预训练模型已经学会了边缘、纹理、形状等通用视觉特征，
我们只需要教会它区分这 15 种装备即可。

为什么迁移学习能解决小样本过拟合:
  - 预训练权重已经包含通用视觉特征（边缘->纹理->物体部件）
  - 只需微调顶层，可训练参数从千万级降到几千
  - 即使每类只有 20 张图，也能学到有意义的分辨能力

运行前安装:
  pip install torch torchvision Pillow
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
# 超参数配置
# ============================================================
IMAGE_SIZE    = 224       # ResNet 标准输入尺寸
BATCH_SIZE    = 16        # 批次大小
EPOCHS        = 30        # 训练轮数（迁移学习收敛快，30 轮足够）
LEARNING_RATE = 0.001     # 学习率
RANDOM_SEED   = 42
TEST_SPLIT    = 0.2       # 测试集比例
NUM_CLASSES   = 15        # 类别数
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练集：数据增强 + 标准化
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试集：只做标准化，不做增强
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================
# 数据集类
# ============================================================
class ImageFolderDataset(Dataset):
    """从文件夹加载图片，目录结构: root_dir/类别名/图片.jpg"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_names = []

        class_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_names = class_dirs

        for label_idx, class_name in enumerate(class_dirs):
            class_dir = os.path.join(root_dir, class_name)
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff', '*.webp']
            for ext in extensions:
                for img_path in glob.glob(os.path.join(class_dir, ext)):
                    self.samples.append((img_path, label_idx))
                for img_path in glob.glob(os.path.join(class_dir, ext.upper())):
                    self.samples.append((img_path, label_idx))

        self.samples = list(set(self.samples))
        print(f"  加载 {len(self.samples)} 张图片, {len(class_dirs)} 个类别: {class_dirs}")

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
# 创建 ResNet18 迁移学习模型
# ============================================================
def create_model(num_classes, freeze_backbone=True):
    """
    基于 ResNet18 的迁移学习模型。

    架构:
      ResNet18 backbone (预训练) -> 全局平均池化 -> FC(512 -> 256 -> num_classes)

    参数:
      num_classes:     分类类别数
      freeze_backbone: True=冻结卷积层, False=全部可训练
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model


# ============================================================
# 训练和评估
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, 100.0 * correct / total, all_preds, all_labels


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  ResNet18 迁移学习 CNN 图片分类器")
    print(f"  运行设备: {DEVICE}")
    print("=" * 65)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 加载数据
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

    # 测试集重新创建，使用无增强的 transform
    test_full = ImageFolderDataset(data_dir, transform=test_transform)
    torch.manual_seed(RANDOM_SEED)
    _, test_dataset = random_split(
        test_full, [n_train, n_test],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n  数据划分:")
    print(f"    训练集: {n_train} 样本")
    print(f"    测试集: {n_test} 样本")
    print(f"    图片尺寸: {IMAGE_SIZE}x{IMAGE_SIZE} RGB")

    # 创建模型
    model = create_model(NUM_CLASSES, freeze_backbone=True)
    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  模型: ResNet18 (预训练)")
    print(f"    总参数: {total_params:,}")
    print(f"    可训练: {trainable:,} (仅顶层 FC)")
    print(f"    冻结: {total_params - trainable:,} (卷积 backbone)")

    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"\n  开始训练...\n")
    print(f"  {'Epoch':<8} {'Train Loss':<12} {'Train Acc':<10} {'Test Loss':<12} {'Test Acc':<10}")
    print(f"  {'-'*55}")

    best_acc = 0.0
    t_start = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        test_loss, test_acc, test_preds, test_labels = evaluate(
            model, test_loader, criterion, DEVICE
        )
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(os.path.dirname(__file__), "best_cnn_model.pth"))

        print(f"  {epoch+1:<8} {train_loss:<12.4f} {train_acc:<10.1f} {test_loss:<12.4f} {test_acc:<10.1f}")

    elapsed = time.time() - t_start

    print(f"\n{'='*65}")
    print(f"  训练完成！耗时 {elapsed:.1f}s")
    print(f"  最佳测试准确率: {best_acc:.2f}%")
    print(f"  模型已保存至: best_cnn_model.pth")

    # 各类别准确率
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    print(f"\n  各类别测试准确率:")
    for cls_idx in range(NUM_CLASSES):
        mask = test_labels == cls_idx
        if np.sum(mask) > 0:
            cls_acc = np.mean(test_preds[mask] == test_labels[mask]) * 100
            name = class_names[cls_idx] if cls_idx < len(class_names) else f"类{cls_idx}"
            bar = "#" * int(cls_acc / 10)
            print(f"     {name:<12} {cls_acc:5.1f}% {bar} ({int(np.sum(mask))}样本)")