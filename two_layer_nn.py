"""
基于 NumPy 的两层全连接神经网络
============================================================
网络结构: 输入层 -> 隐藏层(Sigmoid) -> 输出层(Softmax)
优化手段: Mini-batch SGD + 动量 + L2 正则 + 早停 + 学习率衰减

超参数调优指引:
  - 隐藏层神经元: N_HIDDEN
  - 学习率:        LEARNING_RATE
  - 迭代轮数:      EPOCHS
  - L2 正则强度:   L2_LAMBDA
  - 动量系数:      MOMENTUM
============================================================
"""

import numpy as np
import os
import glob
import time

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ============================================================
# 图片加载工具
# ============================================================
def load_images_from_folders(root_dir, target_size=(28, 28), grayscale=True,
                              normalize=True, verbose=True):
    """
    从文件夹加载图片，自动按子文件夹分类。
    返回: X (numpy数组), y (one-hot标签), class_names (类别名列表)
    """
    if not HAS_PIL and not HAS_CV2:
        raise ImportError("需要安装图片处理库: pip install Pillow")

    class_names = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    if len(class_names) == 0:
        raise ValueError(f"'{root_dir}' 下未找到类别子文件夹")

    n_classes = len(class_names)
    if verbose:
        print(f"发现 {n_classes} 个类别: {class_names}")

    all_images, all_labels = [], []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(root_dir, class_name)
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff', '*.webp']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(class_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(class_dir, ext.upper())))
        image_paths = sorted(set(image_paths))

        if verbose:
            print(f"  {class_name}/: {len(image_paths)} 张图片")

        if len(image_paths) == 0:
            continue

        for img_path in image_paths:
            try:
                if HAS_PIL:
                    img = Image.open(img_path)
                    if grayscale:
                        img = img.convert('L')
                    else:
                        img = img.convert('RGB')
                    img = img.resize(target_size, Image.LANCZOS)
                    arr = np.array(img, dtype=np.float32)
                elif HAS_CV2:
                    if grayscale:
                        arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    else:
                        arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    arr = cv2.resize(arr, target_size)
                    arr = arr.astype(np.float32)

                if normalize:
                    arr = arr / 255.0

                all_images.append(arr.flatten())
                all_labels.append(label_idx)
            except Exception as e:
                print(f"  跳过损坏图片 {img_path}: {e}")

    if len(all_images) == 0:
        raise ValueError("未加载到任何图片")

    X = np.stack(all_images, axis=0)

    if n_classes == 1:
        y = np.array(all_labels, dtype=np.float32).reshape(-1, 1)
    else:
        y = np.zeros((len(all_labels), n_classes), dtype=np.float32)
        y[np.arange(len(all_labels)), all_labels] = 1.0

    if verbose:
        h, w = target_size
        channels = 1 if grayscale else 3
        print(f"\n加载完成: {X.shape[0]} 张图片, 每张 {h}x{w}x{channels} = {X.shape[1]} 个特征")
        print(f"  X shape: {X.shape}, y shape: {y.shape}")
        print(f"  类别映射: {dict(enumerate(class_names))}")

    return X, y, class_names


# ============================================================
# 数据集填写区域
# ============================================================
X, y, CLASS_NAMES = load_images_from_folders(
    root_dir=r"D:\onedrive\桌面\pic",
    target_size=(32, 32),
    grayscale=False,
)


# ============================================================
# 数据增强模块
# ============================================================
def augment_data(X, y, factor=4, noise_std=0.005, flip_prob=0.5, seed=42):
    """
    数据增强：对原始训练数据施加微小扰动，生成更多样本。

    增强策略:
      1. 加高斯噪声 -- 模拟拍照时的轻微曝光差异
      2. 随机翻转 -- 模拟左右镜像
      3. 像素值缩放 -- 模拟亮度变化

    返回:
      X_aug: 增强后的数据, shape (n * factor, d)
      y_aug: 增强后的标签, shape (n * factor, c)
    """
    np.random.seed(seed)
    n, d = X.shape
    side = int(np.sqrt(d))

    X_list, y_list = [X], [y]

    for i in range(factor - 1):
        X_aug = X.copy()

        # 策略1: 加高斯噪声
        noise = np.random.randn(n, d) * noise_std
        X_aug = X_aug + noise

        # 策略2: 随机左右翻转（仅当 d 是正方形像素数时有效）
        if side * side == d:
            for j in range(n):
                if np.random.random() < flip_prob:
                    img_2d = X_aug[j].reshape(side, side)
                    img_2d = np.fliplr(img_2d)
                    X_aug[j] = img_2d.flatten()

        # 策略3: 像素值随机缩放（模拟亮度变化 ±10%）
        brightness = np.random.uniform(0.9, 1.1, (n, 1))
        X_aug = X_aug * brightness
        X_aug = np.clip(X_aug, 0.0, 1.0)

        X_list.append(X_aug)
        y_list.append(y)

    X_aug = np.vstack(X_list)
    y_aug = np.vstack(y_list)

    # 打乱增强后的数据
    indices = np.random.permutation(len(X_aug))
    return X_aug[indices], y_aug[indices]


# ============================================================
# 超参数配置
# ============================================================
N_HIDDEN      = 64        # 隐藏层神经元数
LEARNING_RATE = 0.05      # 初始学习率
EPOCHS        = 2000      # 最大迭代轮数
PRINT_EVERY   = 200       # 打印间隔
RANDOM_SEED   = 42
TEST_SPLIT    = 0.2       # 测试集比例
OUTPUT_ACTIVATION = "auto"

# 防过拟合参数
L2_LAMBDA      = 0.0001   # L2 正则系数
MOMENTUM       = 0.9      # 动量系数
BATCH_SIZE     = 0        # 0=全量梯度下降
LR_DECAY       = 0.0      # 0=不衰减
AUGMENT_FACTOR = 1        # 1=不增强
EARLY_STOP_PATIENCE = 400  # 验证损失连续不降 400 轮则停止


# ============================================================
# 激活函数模块
# ============================================================
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


def softmax(x):
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# ============================================================
# 损失函数模块
# ============================================================
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.sum(y_true * np.log(y_pred)) / m


def cross_entropy_softmax_derivative(y_pred, y_true):
    """Softmax + 交叉熵的联合梯度: dL/dZ = y_pred - y_true"""
    return y_pred - y_true


def mse_loss(y_pred, y_true):
    return np.sum((y_pred - y_true) ** 2) / y_true.shape[0]


def mse_loss_derivative(y_pred, y_true):
    return 2.0 * (y_pred - y_true) / y_true.shape[0]


# ============================================================
# 两层全连接神经网络
# ============================================================
class TwoLayerNN:
    """
    两层全连接神经网络，支持:
      - Mini-batch SGD 训练
      - 动量加速
      - L2 正则化
      - 早停（保留最佳模型）
      - 学习率衰减
    """

    def __init__(self, n_input, n_hidden, n_output, lr=0.05, seed=42,
                 output_activation="softmax", l2_lambda=0.0005, momentum=0.9):
        np.random.seed(seed)
        self.lr = lr
        self.initial_lr = lr
        self.output_activation = output_activation
        self.l2_lambda = l2_lambda
        self.momentum = momentum

        # Xavier 初始化：让每层的输出方差接近输入方差，避免梯度消失/爆炸
        bound1 = np.sqrt(6.0 / (n_input + n_hidden))
        self.W1 = np.random.uniform(-bound1, bound1, (n_input, n_hidden))
        self.b1 = np.zeros((1, n_hidden))

        bound2 = np.sqrt(6.0 / (n_hidden + n_output))
        self.W2 = np.random.uniform(-bound2, bound2, (n_hidden, n_output))
        self.b2 = np.zeros((1, n_output))

        # 动量缓存
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def forward(self, X):
        """前向传播: X -> Z1 -> A1 -> Z2 -> A2"""
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        if self.output_activation == "softmax":
            self.A2 = softmax(self.Z2)
        else:
            self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y):
        """
        反向传播，计算各层梯度。
        链式法则: dL/dW2 = dL/dZ2 * dZ2/dW2 = dL/dZ2 * A1^T
        """
        m = X.shape[0]

        # 输出层梯度
        if self.output_activation == "softmax":
            dZ2 = cross_entropy_softmax_derivative(self.A2, y)
        else:
            dA2 = mse_loss_derivative(self.A2, y)
            dZ2 = dA2 * sigmoid_derivative(self.A2)

        self.dW2 = self.A1.T @ dZ2 / m
        self.db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # 隐藏层梯度
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * sigmoid_derivative(self.A1)
        self.dW1 = X.T @ dZ1 / m
        self.db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    def update_params(self):
        """
        动量梯度下降更新:
          v = momentum * v - lr * dW  (积累历史梯度方向)
          W = W + v                    (沿平滑方向更新)

        动量能加速收敛、抑制震荡，帮助穿越损失曲面的平坦区域。
        """
        self.vW1 = self.momentum * self.vW1 - self.lr * self.dW1
        self.vb1 = self.momentum * self.vb1 - self.lr * self.db1
        self.vW2 = self.momentum * self.vW2 - self.lr * self.dW2
        self.vb2 = self.momentum * self.vb2 - self.lr * self.db2

        self.W1 += self.vW1
        self.b1 += self.vb1
        self.W2 += self.vW2
        self.b2 += self.vb2

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size=32,
              print_every=200, lr_decay=0.98, early_stop_patience=300):
        """
        Mini-batch SGD 训练，含验证监控和早停。

        参数:
          X_train, y_train: 训练数据
          X_val, y_val:     验证数据（用于早停判断）
          batch_size:       Mini-batch 大小
          lr_decay:         每轮学习率衰减因子
          early_stop_patience: 连续不降轮数则停止
        """
        n_train = X_train.shape[0]
        losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        warmup_epochs = 200  # 前 200 轮不触发早停，让网络先学起来

        # 保存最佳权重
        best_W1, best_b1 = self.W1.copy(), self.b1.copy()
        best_W2, best_b2 = self.W2.copy(), self.b2.copy()

        for epoch in range(epochs):
            # Mini-batch 训练: 每轮打乱数据，分批更新
            indices = np.random.permutation(n_train)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)
                self.update_params()

            # 学习率衰减
            self.lr *= lr_decay

            # 定期评估
            if epoch % print_every == 0:
                train_pred = self.forward(X_train)
                if self.output_activation == "softmax":
                    train_loss = cross_entropy_loss(train_pred, y_train)
                else:
                    train_loss = mse_loss(train_pred, y_train)

                val_pred = self.forward(X_val)
                if self.output_activation == "softmax":
                    val_loss = cross_entropy_loss(val_pred, y_val)
                else:
                    val_loss = mse_loss(val_pred, y_val)

                val_pred_class = np.argmax(val_pred, axis=1)
                val_true_class = np.argmax(y_val, axis=1)
                val_acc = np.mean(val_pred_class == val_true_class) * 100

                losses.append((epoch, train_loss, val_loss))
                print(f"  [Epoch {epoch:5d}/{epochs}]  "
                      f"Train Loss={train_loss:.4f}  "
                      f"Val Loss={val_loss:.4f}  "
                      f"Val Acc={val_acc:.1f}%  "
                      f"LR={self.lr:.6f}")

                # 早停检查（仅 warmup 后生效）
                if epoch >= warmup_epochs:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        patience_counter = 0
                        best_W1, best_b1 = self.W1.copy(), self.b1.copy()
                        best_W2, best_b2 = self.W2.copy(), self.b2.copy()
                    else:
                        patience_counter += print_every

                    if patience_counter >= early_stop_patience:
                        print(f"\n  早停触发！验证损失 {early_stop_patience} 轮未改善，"
                              f"最佳轮次: {best_epoch}")
                        self.W1, self.b1 = best_W1, best_b1
                        self.W2, self.b2 = best_W2, best_b2
                        break

        return losses

    def predict(self, X):
        return self.forward(X)


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    t_start = time.time()
    print("=" * 65)
    print("  基于 NumPy 的两层全连接神经网络")
    print("=" * 65)

    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.ndim == 2 and y.ndim == 2
    assert X.shape[0] == y.shape[0]

    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_output = y.shape[1]

    print(f"\n原始数据集: {n_samples} 样本, {n_features} 特征, {n_output} 类别")
    if 'CLASS_NAMES' in dir() and len(CLASS_NAMES) == n_output:
        print(f"  类别: {CLASS_NAMES}")

    if OUTPUT_ACTIVATION == "auto":
        use_activation = "softmax" if n_output > 1 else "sigmoid"
    else:
        use_activation = OUTPUT_ACTIVATION

    print(f"\n网络结构:")
    print(f"  输入层: {n_features} -> 隐藏层: {N_HIDDEN}(Sigmoid) -> 输出层: {n_output}({use_activation})")
    print(f"  总参数: {n_features*N_HIDDEN + N_HIDDEN + N_HIDDEN*n_output + n_output}")

    print(f"\n超参数:")
    print(f"  学习率={LEARNING_RATE} | 动量={MOMENTUM} | 衰减={LR_DECAY}")
    print(f"  L2={L2_LAMBDA} | Batch={BATCH_SIZE} | 增强x{AUGMENT_FACTOR} | 早停={EARLY_STOP_PATIENCE}轮")

    # 步骤1: 打乱并划分
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]
    split_idx = int(n_samples * (1 - TEST_SPLIT))
    X_train_raw, X_test = X[:split_idx], X[split_idx:]
    y_train_raw, y_test = y[:split_idx], y[split_idx:]

    # 步骤2: 数据增强（仅对训练集）
    if AUGMENT_FACTOR > 1:
        print(f"\n数据增强中... (x{AUGMENT_FACTOR})")
        X_train, y_train = augment_data(X_train_raw, y_train_raw,
                                         factor=AUGMENT_FACTOR, seed=RANDOM_SEED)
        print(f"  训练集: {X_train_raw.shape[0]} -> {X_train.shape[0]} 样本")
    else:
        X_train, y_train = X_train_raw, y_train_raw
        print(f"\n数据划分:")
        print(f"  训练集: {X_train.shape[0]} 样本 (无增强)")
        print(f"  测试集: {X_test.shape[0]} 样本")

    # 步骤3: 创建网络
    nn = TwoLayerNN(
        n_input=n_features, n_hidden=N_HIDDEN, n_output=n_output,
        lr=LEARNING_RATE, seed=RANDOM_SEED,
        output_activation=use_activation,
        l2_lambda=L2_LAMBDA, momentum=MOMENTUM
    )
    print(f"\n权重初始化: W1{nn.W1.shape} W2{nn.W2.shape}")

    # 步骤4: 训练
    print(f"\n开始训练...\n")

    batch_size = BATCH_SIZE if BATCH_SIZE > 0 else X_train.shape[0]
    patience = EARLY_STOP_PATIENCE if EARLY_STOP_PATIENCE > 0 else EPOCHS + 1

    losses = nn.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=EPOCHS, batch_size=batch_size,
        print_every=PRINT_EVERY, lr_decay=LR_DECAY if LR_DECAY > 0 else 1.0,
        early_stop_patience=patience
    )

    # 步骤5: 最终评估
    train_pred = nn.predict(X_train)
    test_pred = nn.predict(X_test)

    if use_activation == "softmax":
        train_loss = cross_entropy_loss(train_pred, y_train)
        test_loss = cross_entropy_loss(test_pred, y_test)
    else:
        train_loss = mse_loss(train_pred, y_train)
        test_loss = mse_loss(test_pred, y_test)

    train_pred_raw = nn.predict(X_train_raw)
    test_pred_class = np.argmax(test_pred, axis=1)
    test_true_class = np.argmax(y_test, axis=1)

    if n_output > 1:
        train_pred_raw_class = np.argmax(train_pred_raw, axis=1)
        train_true_raw_class = np.argmax(y_train_raw, axis=1)
        train_acc = np.mean(train_pred_raw_class == train_true_raw_class) * 100
        test_acc = np.mean(test_pred_class == test_true_class) * 100
    else:
        train_acc = np.mean((train_pred_raw >= 0.5).astype(float) == y_train_raw) * 100
        test_acc = np.mean((test_pred >= 0.5).astype(float) == y_test) * 100

    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"训练完成！耗时 {elapsed:.1f}s")
    print(f"  训练集 Loss: {train_loss:.6f}  |  准确率: {train_acc:.2f}%")
    print(f"  测试集 Loss: {test_loss:.6f}  |  准确率: {test_acc:.2f}%")

    # 步骤6: 各类别准确率
    if n_output > 1 and 'CLASS_NAMES' in dir() and len(CLASS_NAMES) == n_output:
        print(f"\n各类别测试准确率:")
        for cls_idx in range(n_output):
            mask = test_true_class == cls_idx
            if np.sum(mask) > 0:
                cls_acc = np.mean(test_pred_class[mask] == test_true_class[mask]) * 100
                print(f"  {CLASS_NAMES[cls_idx]}: {cls_acc:.1f}% ({int(np.sum(mask))}样本)")

    # 步骤7: 抽样对比
    print(f"\n测试集抽样对比:")
    print(f"  {'真实':<14} {'预测':<14} {'结果':<6}")
    print(f"  {'-'*34}")
    show_num = min(15, len(X_test))
    for i in range(show_num):
        tn = CLASS_NAMES[test_true_class[i]] if 'CLASS_NAMES' in dir() else f"类{test_true_class[i]}"
        pn = CLASS_NAMES[test_pred_class[i]] if 'CLASS_NAMES' in dir() else f"类{test_pred_class[i]}"
        flag = "正确" if test_pred_class[i] == test_true_class[i] else "错误"
        print(f"  {tn:<14} {pn:<14} {flag:<6}")

    print(f"\n{'='*65}")
    print(f"当前配置:")
    print(f"  1. 彩色图 32x32 -- 保留颜色区分信息")
    print(f"  2. 隐藏层 {N_HIDDEN}神经元 -- 平衡拟合与泛化")
    print(f"  3. 动量 {MOMENTUM} -- 加速收敛、抑制震荡")
    print(f"  4. L2正则 lambda={L2_LAMBDA} -- 轻微惩罚大权重")
    print(f"  5. 全量梯度下降 -- 稳定梯度方向")
    print(f"  6. 固定学习率 {LEARNING_RATE} -- 避免过早衰减")
    print(f"{'='*65}")