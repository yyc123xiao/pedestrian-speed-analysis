"""
行人检测 + 多目标追踪 + 实时速率计算
=====================================
使用 YOLOv8n 检测行人，IoU+匈牙利算法追踪，3D 空间速度计算。
支持视频文件和摄像头实时输入。
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment


class PersonTracker:
    """多目标行人追踪器，含世界空间速率估算"""

    def __init__(self, fps=30, history_frames=15, max_lost=30,
                 scale_mode="auto", manual_scale=0.02, conf_threshold=0.5):
        self.model = YOLO("yolov8n.pt")
        self.fps = fps
        self.history_frames = history_frames
        self.max_lost = max_lost
        self.scale_mode = scale_mode
        self.manual_scale = manual_scale
        self.conf_threshold = conf_threshold

        self.tracks = {}          # {id: {bbox, centroid, history, lost, speed}}
        self.speed_history = {}   # {id: [(frame_num, speed_m_s), ...]}
        self.next_id = 1
        self.frame_count = 0

    # ================================================================
    #  公共接口
    # ================================================================
    def process_frame(self, frame):
        """处理一帧，返回标注后的帧"""
        self.frame_count += 1

        # YOLO 推理
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            if int(box.cls[0]) != 0:   # person class only
                continue
            if float(box.conf[0]) < self.conf_threshold:  # 过滤低置信度
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append([x1, y1, x2, y2])

        # 数据关联
        self._associate_detections(detections)

        # 速率 + 可视化
        for tid, track in self.tracks.items():
            if track["lost"] > 0:
                continue
            self._update_speed(tid, track)
            self._draw_track(frame, tid, track)

        # 左上角信息
        active = sum(1 for t in self.tracks.values() if t["lost"] == 0)
        cv2.putText(frame, f"Tracking: {active}  |  Frame: {self.frame_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame

    def get_report_data(self):
        """导出所有追踪数据"""
        report = []
        for tid in sorted(self.speed_history.keys()):
            data = self.speed_history[tid]
            if len(data) < 3:
                continue
            frames = [d[0] for d in data]
            speeds = [d[1] for d in data]
            avg = sum(speeds) / len(speeds)
            report.append({
                "id": tid, "frames": frames, "speeds": speeds,
                "avg_speed": avg, "total_frames": len(data),
                "time_seconds": len(data) / self.fps if self.fps > 0 else 0,
            })
        return report

    # ================================================================
    #  数据关联 — 匈牙利算法 + IoU
    # ================================================================
    def _associate_detections(self, detections):
        active_ids = [tid for tid, t in self.tracks.items() if t["lost"] <= self.max_lost]

        if not detections:
            for tid in self.tracks:
                self.tracks[tid]["lost"] += 1
            self._cleanup()
            return

        if not active_ids:
            for det in detections:
                self._create_track(det)
            return

        n_tracks, n_dets = len(active_ids), len(detections)
        cost = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for i, tid in enumerate(active_ids):
            tb = self.tracks[tid]["bbox"]
            for j, det in enumerate(detections):
                cost[i, j] = 1.0 - self._iou(tb, det)

        row_idx, col_idx = linear_sum_assignment(cost)

        matched_tracks, matched_dets = set(), set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < 0.7:  # IoU > 0.3
                tid, det = active_ids[r], detections[c]
                self._update_track(tid, det)
                matched_tracks.add(tid)
                matched_dets.add(c)

        for tid in active_ids:
            if tid not in matched_tracks:
                self.tracks[tid]["lost"] += 1

        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._create_track(det)

        self._cleanup()

    def _create_track(self, det):
        x1, y1, x2, y2 = det
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        tid = self.next_id
        self.tracks[tid] = {
            "bbox": [x1, y1, x2, y2],
            "centroid": (cx, cy),
            "history": deque(maxlen=self.history_frames),
            "lost": 0,
            "speed": 0.0,
        }
        self.tracks[tid]["history"].append((cx, cy, self.frame_count))
        self.speed_history[tid] = []
        self.next_id += 1

    def _update_track(self, tid, det):
        x1, y1, x2, y2 = det
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        self.tracks[tid]["bbox"] = [x1, y1, x2, y2]
        self.tracks[tid]["centroid"] = (cx, cy)
        self.tracks[tid]["lost"] = 0
        self.tracks[tid]["history"].append((cx, cy, self.frame_count))

    # ================================================================
    #  3D 空间速度计算
    # ================================================================
    def _update_speed(self, tid, track):
        hist = list(track["history"])
        if len(hist) < 3:
            track["speed"] = 0.0
            return

        x0, y0, f0 = hist[0]
        x1, y1, f1 = hist[-1]
        df = f1 - f0
        if df == 0:
            track["speed"] = 0.0
            return

        pixel_dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        dt = df / self.fps

        # 比例尺
        if self.scale_mode == "manual":
            scale = self.manual_scale
        else:
            _, by1, _, by2 = track["bbox"]
            h = by2 - by1
            scale = 0.02 if h < 10 else 1.70 / h  # 假设身高 1.7m

        speed = pixel_dist * scale / dt if dt > 0 else 0.0
        track["speed"] = speed

        # 记录全视频速度历史
        if tid not in self.speed_history:
            self.speed_history[tid] = []
        self.speed_history[tid].append((self.frame_count, speed))

    # ================================================================
    #  可视化
    # ================================================================
    def _draw_track(self, frame, tid, track):
        x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
        speed = track["speed"]

        # 慢→绿，快→红（10 m/s 封顶）
        ratio = min(speed / 10.0, 1.0)
        color = (0, int(255 * (1 - ratio)), int(255 * ratio))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{tid}  {speed:.1f} m/s"
        lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        ly = y1 - 8 if y1 > 28 else y1 + lh + 8
        cv2.rectangle(frame, (x1, ly - lh - 4), (x1 + lw + 6, ly + 4), color, -1)
        cv2.putText(frame, label, (x1 + 3, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 轨迹线
        hist = list(track["history"])
        for i in range(1, len(hist)):
            p1 = (int(hist[i - 1][0]), int(hist[i - 1][1]))
            p2 = (int(hist[i][0]), int(hist[i][1]))
            cv2.line(frame, p1, p2, color, 2)

    # ================================================================
    #  工具函数
    # ================================================================
    def _cleanup(self):
        to_remove = [tid for tid, t in self.tracks.items() if t["lost"] > self.max_lost]
        for tid in to_remove:
            del self.tracks[tid]

    def _iou(self, box_a, box_b):
        xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
        xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
