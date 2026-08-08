"""
图片预缓存工具
============================================================
把图片批量转成 numpy 数组，运行一次后 predict_image.py 自动使用缓存，
推理速度 < 5ms。

用法: python precache_images.py
============================================================
"""

import os
import sys
import glob
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ============================================================
# 配置
# ============================================================
IMAGE_DIR = r"D:\onedrive\桌面\pic"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "image_cache")

# 与训练时完全一致的预处理，保证推理结果准确
fast_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

acc_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    print("=" * 55)
    print("  图片预缓存工具")
    print("=" * 55)

    if not os.path.exists(IMAGE_DIR):
        print(f"\n图片目录不存在: {IMAGE_DIR}")
        sys.exit(1)

    # 收集所有图片
    images = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff', '*.webp']:
        for pattern in [ext, ext.upper()]:
            images.extend(glob.glob(os.path.join(IMAGE_DIR, '**', pattern), recursive=True))
    images = sorted(set(images))

    if not images:
        print(f"\n未找到图片: {IMAGE_DIR}")
        sys.exit(1)

    print(f"\n找到 {len(images)} 张图片")

    # 清理旧缓存
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"缓存目录: {CACHE_DIR}")

    t_start = time.time()
    processed = 0

    for img_path in images:
        try:
            image = Image.open(img_path).convert('RGB')

            fast_arr = fast_transform(image).numpy()
            acc_arr = acc_transform(image).numpy()

            rel_path = os.path.relpath(img_path, IMAGE_DIR)
            base_name = os.path.splitext(rel_path)[0]

            fast_file = os.path.join(CACHE_DIR, base_name + '_fast.npy')
            acc_file = os.path.join(CACHE_DIR, base_name + '_acc.npy')

            os.makedirs(os.path.dirname(fast_file), exist_ok=True)
            np.save(fast_file, fast_arr)
            np.save(acc_file, acc_arr)

            processed += 1
            if processed % 50 == 0:
                print(f"  进度: {processed}/{len(images)}")
        except Exception as e:
            print(f"  跳过 {img_path}: {e}")

    elapsed = time.time() - t_start
    print(f"\n完成！{processed} 张图片 -> {elapsed:.1f}s")
    print(f"  缓存位置: {CACHE_DIR}")
    print(f"\n提示: 现在运行 predict_image.py 将自动使用缓存")


if __name__ == "__main__":
    main()