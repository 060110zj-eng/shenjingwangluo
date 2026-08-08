"""
=============================================================================
  基于 PyTorch + ResNet18 迁移学习的 CNN 图片分类器
  =====================================================================
  策略: 使用在 ImageNet 上预训练的 ResNet18，冻结底层卷积层，
        只训练最后的全连接分类层。
        预训练模型已经学会了"识别边缘/纹理/形状"的能力，
        我们只需教会它"区分这 15 种装备"。
  
  为什么迁移学习能解决过拟合:
    - 预训练权重已包含通用视觉特征（边缘→纹理→物体部件）
    - 我们只需微调顶层，参数量从 1100 万降到几千
    - 即使每类只有 20 张图也能学到有意义的分辨能力
  
  运行前请安装:
    pip install torch torchvision Pillow
  
  作者: AI Assistant
  日期: 2026-08-06
=============================================================================
"""

import os
import glob
import numpy as np
import time

# ---------- 深度学习框架 ----------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image

# ===========================================================================
#  一、超参数配置
# ===========================================================================
IMAGE_SIZE    = 224      # ResNet 标准输入尺寸
BATCH_SIZE    = 16       # 批次大小
EPOCHS        = 30       # 训练轮数（迁移学习收敛快，30轮足够）
LEARNING_RATE = 0.001    # 学习率
RANDOM_SEED   = 42
TEST_SPLIT    = 0.2      # 测试集比例
NUM_CLASSES   = 15       # 类别数
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强（训练集）和 标准化（训练+测试）
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),       # 随机水平翻转
    transforms.RandomRotation(degrees=15),         # 随机旋转 ±15°
    transforms.ColorJitter(brightness=0.2,         # 亮度抖动
                           contrast=0.2,           # 对比度抖动
                           saturation=0.2),        # 饱和度抖动
    transforms.ToTensor(),                         # 转 Tensor + 归一化到 [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet 均值
                         std=[0.229, 0.224, 0.225])    # ImageNet 标准差
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ===========================================================================
#  二、自定义数据集类
# ===========================================================================
class ImageFolderDataset(Dataset):
    """
    从文件夹加载图片数据集。
    期望结构: root_dir/类别名/图片.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_names = []
        
        # 扫描所有类别文件夹
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
        
        # 去重
        self.samples = list(set(self.samples))
        print(f"📁 加载 {len(self.samples)} 张图片, {len(class_dirs)} 个类别: {class_dirs}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # 损坏图片返回黑色图
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ===========================================================================
#  三、创建 ResNet18 迁移学习模型
# ===========================================================================
def create_model(num_classes, freeze_backbone=True):
    """
    创建基于 ResNet18 的迁移学习模型。
    
    架构:
      ResNet18 backbone (预训练) → 全局平均池化 → FC(512→num_classes)
    
    参数:
        num_classes:     分类类别数
        freeze_backbone: True=冻结卷积层, False=全部可训练
    """
    # 加载预训练 ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    if freeze_backbone:
        # 冻结所有卷积层参数（不更新梯度）
        for param in model.parameters():
            param.requires_grad = False
    
    # 替换最后的全连接层
    in_features = model.fc.in_features  # ResNet18 是 512
    model.fc = nn.Sequential(
        nn.Dropout(0.3),                          # Dropout 防过拟合
        nn.Linear(in_features, 256),              # 中间层
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)               # 输出层
    )
    
    return model


# ===========================================================================
#  四、训练和评估函数
# ===========================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估模型"""
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


# ===========================================================================
#  五、主程序
# ===========================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  ResNet18 迁移学习 — CNN 图片分类器")
    print(f"  运行设备: {DEVICE}")
    print("=" * 65)
    
    # ---------- 设置随机种子 ----------
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # ---------- 加载数据 ----------
    data_dir = r"D:\onedrive\桌面\pic"
    full_dataset = ImageFolderDataset(data_dir, transform=train_transform)
    class_names = full_dataset.class_names
    
    # 划分训练集/测试集
    n_total = len(full_dataset)
    n_test = int(n_total * TEST_SPLIT)
    n_train = n_total - n_test
    
    train_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    # 测试集用不同的 transform（无数据增强）
    test_dataset.dataset.transform = test_transform
    # 注意：random_split 返回 Subset，需要单独处理 transform
    # 更简单的方法：重新创建测试集 Dataset
    test_full = ImageFolderDataset(data_dir, transform=test_transform)
    # 用相同的随机种子划分，保证测试集一致
    torch.manual_seed(RANDOM_SEED)
    _, test_dataset = random_split(
        test_full, [n_train, n_test],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n📦 数据划分:")
    print(f"   训练集: {n_train} 样本 ({BATCH_SIZE} batch)")
    print(f"   测试集: {n_test} 样本")
    print(f"   图片尺寸: {IMAGE_SIZE}×{IMAGE_SIZE} RGB")
    
    # ---------- 创建模型 ----------
    model = create_model(NUM_CLASSES, freeze_backbone=True)
    model = model.to(DEVICE)
    
    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🔧 模型: ResNet18 (预训练)")
    print(f"   总参数: {total_params:,}")
    print(f"   可训练: {trainable:,} (仅顶层 FC)")
    print(f"   冻结层: {total_params - trainable:,} (卷积 backbone)")
    
    # ---------- 训练 ----------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print(f"\n🚀 开始训练...\n")
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
            # 保存最佳模型
            torch.save(model.state_dict(), 
                       os.path.join(os.path.dirname(__file__), "best_cnn_model.pth"))
        
        print(f"  {epoch+1:<8} {train_loss:<12.4f} {train_acc:<10.1f} {test_loss:<12.4f} {test_acc:<10.1f}")
    
    elapsed = time.time() - t_start
    
    # ---------- 最终结果 ----------
    print(f"\n{'='*65}")
    print(f"✅ 训练完成！耗时 {elapsed:.1f}s")
    print(f"   最佳测试准确率: {best_acc:.2f}%")
    print(f"   模型已保存至: best_cnn_model.pth")
    
    # ---------- 各类别准确率 ----------
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    
    print(f"\n📋 各类别测试准确率:")
    for cls_idx in range(NUM_CLASSES):
        mask = test_labels == cls_idx
        if np.sum(mask) > 0:
            cls_acc = np.mean(test_preds[mask] == test_labels[mask]) * 100
            name = class_names[cls_idx] if cls_idx < len(class_names) else f"类{cls_idx}"
            bar = "█" * int(cls_acc / 10)
            print(f"   {name:<12} {cls_acc:5.1f}% {bar} ({int(np.sum(mask))}样本)")
