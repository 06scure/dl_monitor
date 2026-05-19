#!/usr/bin/env python3
"""测试脚本 — 打印所有接收到的命令行参数"""

import sys
import time
from datetime import datetime

print("=" * 60)
print("  test.py — 命令行参数测试")
print("=" * 60)
print()
print(f"Python 路径: {sys.executable}")
print(f"脚本路径:   {sys.argv[0]}")
print(f"参数数量:   {len(sys.argv) - 1}")
print()

if len(sys.argv) > 1:
    for i, arg in enumerate(sys.argv[1:], start=1):
        print(f"  arg[{i}]: {arg}")
else:
    print("  (未传入额外参数)")

print()
print(f"开始时间: {datetime.now()}")
print("--- 模拟训练过程 ---")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

if tqdm:
    # 模拟 5 个 epoch，每个 epoch 包含 100 个 batch 的训练
    for epoch in range(1, 6):
        print(f"\nEpoch {epoch}/5")
        for batch in tqdm(range(100), desc=f"  Training", unit="batch", ncols=80):
            time.sleep(0.02)
        for batch in tqdm(range(50), desc=f"  Validation", unit="batch", ncols=80):
            time.sleep(0.02)
        print(f"  Epoch {epoch} - loss: {1.0 / epoch:.4f} - acc: {0.5 + epoch * 0.1:.2f}")
else:
    print("(tqdm 未安装，使用普通输出)")
    for epoch in range(1, 6):
        print(f"Epoch {epoch}/5 - loss: {1.0 / epoch:.4f} - acc: {0.5 + epoch * 0.1:.2f}")
        time.sleep(0.5)

print(f"\n结束时间: {datetime.now()}")
print("训练完成!")
