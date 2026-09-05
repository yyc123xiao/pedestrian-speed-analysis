"""
生成包含大量误检的测试视频
策略：在真人视频上叠加 YOLO 容易误判为人形的干扰物
- 树干/柱状物（竖长矩形）
- 模糊人影（低分辨率的人形轮廓）
- 背包/包裹（块状物体）
- 随机噪点区域
"""
import cv2
import numpy as np
import random

# ── 读取真人视频作为底 ──
cap = cv2.VideoCapture("test_video.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("test_video_with_noise.mp4", fourcc, fps, (w, h))

# ── 干扰物模板 ──
# 1. 竖长矩形（模拟树干/柱子）
def draw_pillar(frame, x, y, bw, bh, alpha=0.7):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (80, 60, 40), -1)
    # 加纹理
    for _ in range(bh // 2):
        rx = x + random.randint(0, bw)
        ry = y + random.randint(0, bh)
        cv2.line(overlay, (rx, ry), (x + bw//2 + random.randint(-3, 3), ry + 10), (60, 40, 20), 2)
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

# 2. 模糊人形轮廓
def draw_blurry_human(frame, cx, cy, scale=1.0):
    pts = np.array([
        [cx - 10*scale, cy - 40*scale],   # 头顶
        [cx - 5*scale,  cy - 20*scale],   # 肩左
        [cx - 15*scale, cy],              # 手左
        [cx - 8*scale,  cy],              # 胯左
        [cx - 8*scale,  cy + 40*scale],   # 脚左
        [cx + 8*scale,  cy + 40*scale],   # 脚右
        [cx + 8*scale,  cy],              # 胯右
        [cx + 15*scale, cy],              # 手右
        [cx + 5*scale,  cy - 20*scale],   # 肩右
    ], np.int32)
    overlay = frame.copy()
    color = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 180))
    cv2.fillPoly(overlay, [pts], color)
    result = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
    # 高斯模糊模拟远处模糊人影
    result = cv2.GaussianBlur(result, (9, 9), 3)
    # 只在人形区域模糊
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
    return (frame * (1 - mask_3ch) + result * mask_3ch).astype(np.uint8)

# 3. 块状物体（背包/石头）
def draw_blob(frame, x, y, r):
    overlay = frame.copy()
    cv2.ellipse(overlay, (x, y), (r, r//2), 0, 0, 360, (60, 55, 50), -1)
    return cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)


frame_idx = 0
interference_phases = [
    # (start_frame, end_frame, type, count)
    (0,   150, "pillar",  3),    # 第一阶段：柱子
    (150, 280, "blurry",  4),    # 第二阶段：模糊人影
    (280, 400, "blob",    5),    # 第三阶段：块状物体
    (400, 500, "mixed",   6),    # 第四阶段：混合
    (500, 596, "pillar",  2),    # 第五阶段：少量柱子
]

print("Generating noisy test video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 确定当前阶段的干扰物
    current_phase = None
    for start, end, ptype, count in interference_phases:
        if start <= frame_idx < end:
            current_phase = (ptype, count)
            break

    if current_phase:
        ptype, count = current_phase
        for _ in range(count):
            if ptype == "pillar":
                bw = random.randint(12, 30)
                bh = random.randint(80, 180)
                x = random.randint(0, w - bw)
                y = random.randint(h//4, h - bh)
                frame = draw_pillar(frame, x, y, bw, bh, random.uniform(0.3, 0.7))
            elif ptype == "blurry":
                cx = random.randint(40, w - 40)
                cy = random.randint(120, h - 100)
                frame = draw_blurry_human(frame, cx, cy, random.uniform(0.4, 1.2))
            elif ptype == "blob":
                r = random.randint(20, 50)
                x = random.randint(r, w - r)
                y = random.randint(h//3, h - r)
                frame = draw_blob(frame, x, y, r)
            elif ptype == "mixed":
                rtype = random.choice(["pillar", "blurry", "blob"])
                if rtype == "pillar":
                    bw = random.randint(12, 25)
                    bh = random.randint(70, 150)
                    x = random.randint(0, w - bw)
                    y = random.randint(h//4, h - bh)
                    frame = draw_pillar(frame, x, y, bw, bh, random.uniform(0.3, 0.6))
                elif rtype == "blurry":
                    cx = random.randint(40, w - 40)
                    cy = random.randint(120, h - 100)
                    frame = draw_blurry_human(frame, cx, cy, random.uniform(0.3, 1.0))
                else:
                    r = random.randint(15, 45)
                    x = random.randint(r, w - r)
                    y = random.randint(h//3, h - r)
                    frame = draw_blob(frame, x, y, r)

    out.write(frame)
    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"  Frame {frame_idx}/{total}")

cap.release()
out.release()
print(f"\nDone! Output: test_video_with_noise.mp4")
print(f"Total: {frame_idx} frames, {frame_idx/fps:.1f}s")
print(f"Interference types: pillar (trunk-like), blurry human, blob (bag-like)")
