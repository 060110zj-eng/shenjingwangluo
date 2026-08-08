"""
图片识别工具 -- ONNX 极速版
============================================================
策略:
  1. 内存缓存 numpy 数组，加载几乎不耗时
  2. ONNX Runtime SqueezeNet 推理，约 1ms
  3. 置信度 >= 90% 直接出结果
  4. 置信度 < 90% 自动求助 ResNet18 精准复核

用法:
  python predict_image.py 图片路径      # 单张识别
  python predict_image.py              # 交互模式
============================================================
"""

import os
import sys
import time
import numpy as np
import cv2
import onnxruntime as ort

# ============================================================
# 配置
# ============================================================
ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), "squeezenet_fp32.onnx")
ONNX_ACC_PATH   = os.path.join(os.path.dirname(__file__), "resnet18_fp32.onnx")
ACC_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "best_cnn_model.pth")
CACHE_DIR       = os.path.join(os.path.dirname(__file__), "image_cache")
IMAGE_DIR       = r"D:\onedrive\桌面\pic"

CLASS_NAMES = [
    'M救护车', 'a枪', 'b手雷', 'c匕首', 'd警棍',
    'e消防斧', 'f急救包', 'g手电筒', 'h对讲机', 'i防弹背心',
    'j望远镜', 'k头盔', 'l消防车', 'n装甲车', 'o摩托车'
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
# 预处理：优先从内存缓存读取，回退到 OpenCV 解码
# ============================================================
_ram_cache = {}


def _init_ram_cache():
    """启动时把所有 .npy 缓存文件一次性读入内存，之后不再访问磁盘"""
    global _ram_cache
    if not os.path.exists(CACHE_DIR):
        return 0
    count = 0
    for root, _, files in os.walk(CACHE_DIR):
        for f in files:
            if f.endswith(('_fast.npy', '_acc.npy')):
                path = os.path.join(root, f)
                folder = os.path.basename(root)
                key = (folder, f)
                try:
                    _ram_cache[key] = np.load(path)
                    count += 1
                except Exception:
                    pass
    return count


def _get_cache_path(image_path, size):
    """根据图片路径查找对应的缓存数据"""
    folder = os.path.basename(os.path.dirname(image_path))
    name = os.path.splitext(os.path.basename(image_path))[0]
    suffix = '_fast.npy' if size == 112 else '_acc.npy'
    filename = name + suffix

    # 先查内存缓存
    key = (folder, filename)
    if key in _ram_cache:
        return key

    # 回退到磁盘
    cache_path = os.path.join(CACHE_DIR, folder, filename)
    return cache_path if os.path.exists(cache_path) else None


def preprocess_fast(image_path):
    """返回 (1, 3, 112, 112) 的 float32 numpy 数组"""
    cache = _get_cache_path(image_path, 112)
    if cache is not None:
        if isinstance(cache, tuple):
            arr = _ram_cache[cache]
        else:
            arr = np.load(cache)
        return arr[np.newaxis, :]

    # 没有缓存，用 OpenCV 实时解码
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)
    return img[np.newaxis, :]


def preprocess_accurate(image_path):
    """返回 (1, 3, 224, 224) 的 float32 numpy 数组"""
    cache = _get_cache_path(image_path, 224)
    if cache is not None:
        if isinstance(cache, tuple):
            arr = _ram_cache[cache]
        else:
            arr = np.load(cache)
        return arr[np.newaxis, :]

    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)
    return img[np.newaxis, :]


# ============================================================
# 模型加载
# ============================================================
def load_onnx(path):
    """加载 ONNX Runtime 模型，开启全部图优化，4 线程推理"""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, opts, providers=['CPUExecutionProvider'])


# ============================================================
# 推理
# ============================================================
def infer_onnx(session, arr):
    """ONNX 推理，返回 (类别名, 置信度百分比)"""
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: arr})[0]
    # 数值稳定的 softmax
    output = output - np.max(output, axis=1, keepdims=True)
    exp = np.exp(output)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    idx = np.argmax(probs, axis=1)[0]
    return CLASS_NAMES[idx], probs[0, idx] * 100


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  图片识别器 (ONNX)")
    print("=" * 50)

    # 把缓存文件全部加载到内存
    print(f"\n  预加载缓存到内存...")
    n_cached = _init_ram_cache()
    if n_cached:
        total_mb = sum(a.nbytes for a in _ram_cache.values()) // 1024 // 1024
        print(f"  已加载 {n_cached} 个缓存文件 ({total_mb}MB)")

    has_fast = os.path.exists(ONNX_MODEL_PATH)
    has_acc  = os.path.exists(ONNX_ACC_PATH) or os.path.exists(ACC_MODEL_PATH)

    if not has_fast and not has_acc:
        print(f"\n  未找到模型文件，请先运行 export_onnx.py")
        sys.exit(1)

    fast_session = acc_session = None

    if has_fast:
        print(f"\n  加载极速模型 (SqueezeNet)...")
        fast_session = load_onnx(ONNX_MODEL_PATH)
        input_name = fast_session.get_inputs()[0].name
        for _ in range(5):
            fast_session.run(None, {input_name: np.random.randn(1, 3, 112, 112).astype(np.float32)})

    if os.path.exists(ONNX_ACC_PATH):
        print(f"  加载精准模型 (ResNet18)...")
        acc_session = load_onnx(ONNX_ACC_PATH)
        input_name = acc_session.get_inputs()[0].name
        for _ in range(3):
            acc_session.run(None, {input_name: np.random.randn(1, 3, 224, 224).astype(np.float32)})

    if has_fast and acc_session:
        print(f"  就绪 (双模型协作)\n")
    elif has_fast:
        print(f"  就绪 (极速模式)\n")
    else:
        print(f"  就绪 (精准模式)\n")

    # ----- 命令行参数模式 -----
    if len(sys.argv) > 1:
        image_path = sys.argv[1].strip('"').strip("'")
        if not os.path.exists(image_path):
            print(f"  文件不存在: {image_path}")
            sys.exit(1)

        total_start = time.perf_counter()

        if has_fast:
            t0 = time.perf_counter()
            arr = preprocess_fast(image_path)
            prep_time = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            fast_class, fast_conf = infer_onnx(fast_session, arr)
            infer_time = (time.perf_counter() - t0) * 1000

            if fast_conf >= 90:
                total_time = (time.perf_counter() - total_start) * 1000
                print(f"  结果: {fast_class}")
                print(f"  置信度: {fast_conf:.1f}%")
                print(f"  加载: {prep_time:.1f}ms + 推理: {infer_time:.1f}ms")
                print(f"  总耗时: {total_time:.1f}ms")
                sys.exit(0)
            else:
                need_verify = True
        else:
            need_verify = True

        if need_verify and acc_session:
            t0 = time.perf_counter()
            arr = preprocess_accurate(image_path)
            acc_prep_time = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            acc_class, acc_conf = infer_onnx(acc_session, arr)
            acc_infer_time = (time.perf_counter() - t0) * 1000

            total_time = (time.perf_counter() - total_start) * 1000
            flag = "[高置信]" if acc_conf >= 90 else "[低置信]"
            print(f"  {flag} {acc_class}")
            print(f"  置信度: {acc_conf:.1f}%")
            if has_fast:
                print(f"  极速模型猜测: {fast_class} ({fast_conf:.1f}%)")
                print(f"  极速: {prep_time:.1f}+{infer_time:.1f}ms -> 精准: {acc_prep_time:.1f}+{acc_infer_time:.1f}ms")
            print(f"  总耗时: {total_time:.1f}ms")
        elif need_verify:
            total_time = (time.perf_counter() - total_start) * 1000
            print(f"  [低置信] {fast_class} ({fast_conf:.1f}%)")
            print(f"  耗时: {total_time:.1f}ms")

        sys.exit(0)

    # ----- 交互模式 -----
    print(f"类别: {CLASS_NAMES}\n")
    print("输入图片路径识别，输入 q 退出")
    print("-" * 45)

    while True:
        try:
            user_input = input("路径 > ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print("退出")
            break

        if user_input.lower() in ('q', 'quit', 'exit', ''):
            print("退出")
            break

        if not os.path.exists(user_input):
            print(f"文件不存在\n")
            continue

        try:
            total_start = time.perf_counter()

            if has_fast:
                t0 = time.perf_counter()
                arr = preprocess_fast(user_input)
                prep_time = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                fast_class, fast_conf = infer_onnx(fast_session, arr)
                infer_time = (time.perf_counter() - t0) * 1000

                if fast_conf >= 90:
                    total_time = (time.perf_counter() - total_start) * 1000
                    print(f"  {fast_class}  ({fast_conf:.1f}%, {total_time:.1f}ms)\n")
                    continue
                else:
                    need_verify = True
            else:
                need_verify = True

            if need_verify and acc_session:
                t0 = time.perf_counter()
                arr = preprocess_accurate(user_input)
                acc_prep_time = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                acc_class, acc_conf = infer_onnx(acc_session, arr)
                acc_infer_time = (time.perf_counter() - t0) * 1000

                total_time = (time.perf_counter() - total_start) * 1000
                flag = "[高置信]" if acc_conf >= 90 else "[低置信]"
                print(f"  {flag} {acc_class} ({acc_conf:.1f}%, {total_time:.1f}ms)"
                      f"  [极速: {fast_class} {fast_conf:.1f}%]\n")
            elif need_verify:
                total_time = (time.perf_counter() - total_start) * 1000
                print(f"  [低置信] {fast_class} ({fast_conf:.1f}%, {total_time:.1f}ms)\n")
        except Exception as e:
            print(f"失败: {e}\n")