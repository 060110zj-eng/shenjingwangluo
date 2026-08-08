"""
SqueezeNet 模型导出为 ONNX 格式
============================================================
1. PyTorch -> ONNX
2. ONNX -> INT8 量化（可选）
3. 速度对比
============================================================
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "best_cnn_model_fast.pth")
ONNX_FP32  = os.path.join(BASE_DIR, "squeezenet_fp32.onnx")
ONNX_INT8  = os.path.join(BASE_DIR, "squeezenet_int8.onnx")

# ============================================================
# 1. 加载 PyTorch 模型
# ============================================================
print("加载 SqueezeNet...")
state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model = models.squeezenet1_0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Conv2d(512, 15, kernel_size=1),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1))
)
model.load_state_dict(state_dict)
model.eval()

# ============================================================
# 2. 导出 ONNX
# ============================================================
print("导出 ONNX...")
dummy = torch.randn(1, 3, 112, 112)
torch.onnx.export(
    model, dummy, ONNX_FP32,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=14,
)
print(f"  -> {ONNX_FP32}")

# ============================================================
# 3. ONNX INT8 量化（可选）
# ============================================================
print("INT8 量化...")
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(ONNX_FP32, ONNX_INT8, weight_type=QuantType.QInt8)
    print(f"  -> {ONNX_INT8}")
except Exception as e:
    print(f"  量化失败: {e}，使用 FP32")
    ONNX_INT8 = ONNX_FP32

# ============================================================
# 4. 基准测试
# ============================================================
import onnxruntime as ort

test_input = np.random.randn(1, 3, 112, 112).astype(np.float32)


def benchmark(session, label, warmup=20, runs=100):
    input_name = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {input_name: test_input})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: test_input})
        times.append((time.perf_counter() - t0) * 1000)

    avg = np.mean(times)
    med = np.median(times)
    p99 = np.percentile(times, 99)
    print(f"  {label:12s}  avg={avg:.1f}ms  med={med:.1f}ms  p99={p99:.1f}ms  "
          f"min={np.min(times):.1f}ms  max={np.max(times):.1f}ms")
    return avg, med, p99


print("\nONNX Runtime 基准:")
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.intra_op_num_threads = 4
sess_opts.inter_op_num_threads = 1

for ep in ['CPUExecutionProvider']:
    try:
        sess = ort.InferenceSession(ONNX_FP32, sess_opts, providers=[ep])
        benchmark(sess, f"FP32 {ep}")
    except Exception as e:
        print(f"  失败 {ep}: {e}")

if os.path.exists(ONNX_INT8) and ONNX_INT8 != ONNX_FP32:
    try:
        sess = ort.InferenceSession(ONNX_INT8, sess_opts, providers=['CPUExecutionProvider'])
        benchmark(sess, "INT8")
    except Exception as e:
        print(f"  INT8 失败: {e}")

print("\n单线程模式:")
sess_opts.intra_op_num_threads = 1
sess = ort.InferenceSession(ONNX_FP32, sess_opts, providers=['CPUExecutionProvider'])
benchmark(sess, "FP32 (1线程)")

print("\n完成")