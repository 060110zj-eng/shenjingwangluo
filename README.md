# 神经网络图片分类器

基于 CNN 的 15 种装备图片分类器，支持多种推理方案。

## 模型方案

| 模型 | 文件 | 特点 |
|------|------|------|
| ResNet18（迁移学习） | `cnn_classifier.py` | 高精度，45MB |
| SqueezeNet（极速版） | `fast_cnn.py` | 推理 2-4ms，3MB |
| ONNX 推理 | `predict_image.py` | 极速 1ms，自动级联 ResNet18 复核 |

## 环境安装

```bash
pip install torch torchvision Pillow numpy opencv-python onnxruntime
```

## 使用方法

### 训练模型

```bash
python cnn_classifier.py      # 训练 ResNet18
python fast_cnn.py             # 训练 SqueezeNet 极速模型
```

### 图片识别

```bash
python predict_image.py 图片路径      # 单张识别
python predict_image.py                # 交互模式
```

### 导出 ONNX

```bash
python export_onnx.py
```

## 推理流程

1. SqueezeNet ONNX 极速推理（≈1ms）
2. 置信度 ≥ 90% → 直接输出结果
3. 置信度 < 90% → 自动求助 ResNet18 精准复核

## 项目结构

```
├── cnn_classifier.py         # ResNet18 迁移学习训练
├── fast_cnn.py               # SqueezeNet 极速训练
├── two_layer_nn.py           # NumPy 两层神经网络
├── predict_image.py          # ONNX 推理入口
├── export_onnx.py            # 导出 ONNX 模型
├── precache_images.py        # 图片缓存预处理
├── best_cnn_model.pth        # ResNet18 训练权重
├── best_cnn_model_fast.pth   # SqueezeNet 训练权重
├── squeezenet_fp32.onnx      # SqueezeNet ONNX 模型
├── resnet18_fp32.onnx        # ResNet18 ONNX 模型
└── image_cache/              # 图片缓存目录
```