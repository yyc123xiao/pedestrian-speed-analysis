"""
行人速率分析工具 - GUI
支持视频文件 + 摄像头实时模式
分析结束后自动生成速度报告和曲线图
"""

import os, sys, threading, cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from person_tracker import PersonTracker


class SpeedApp:
    def __init__(self, root, video_path=None):
        self.root = root
        self.root.title("行人速率分析工具")
        self.root.geometry("640x480")
        self.root.resizable(False, False)

        self.tracker = None
        self.running = False
        self.thread = None
        self.source_type = "file"  # "file" or "camera"

        self._build_ui()

        # 如果传了视频路径，自动填入并开始分析
        if video_path and os.path.isfile(video_path):
            self.video_var.set(video_path)
            self.src_var.set("file")
            self._toggle_source()
            self.root.after(500, self._start)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=14)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="行人检测 & 速率分析工具",
                  font=("Microsoft YaHei", 14, "bold")).pack(pady=(0, 12))

        # ---- 输入源：文件 / 摄像头 ----
        src_frame = ttk.LabelFrame(main, text="输入源", padding=8)
        src_frame.pack(fill=tk.X, pady=2)

        src_row = ttk.Frame(src_frame)
        src_row.pack(fill=tk.X)
        self.src_var = tk.StringVar(value="file")
        ttk.Radiobutton(src_row, text="视频文件", variable=self.src_var,
                        value="file", command=self._toggle_source).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(src_row, text="摄像头", variable=self.src_var,
                        value="camera", command=self._toggle_source).pack(side=tk.LEFT)

        # 文件路径
        self.file_row = ttk.Frame(src_frame)
        self.file_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(self.file_row, text="文件:", width=5).pack(side=tk.LEFT)
        self.video_var = tk.StringVar()
        ttk.Entry(self.file_row, textvariable=self.video_var, width=45).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.browse_btn = ttk.Button(self.file_row, text="浏览", command=self._browse_video)
        self.browse_btn.pack(side=tk.LEFT)

        # 摄像头
        self.cam_row = ttk.Frame(src_frame)
        ttk.Label(self.cam_row, text="摄像头编号 (0=默认):", width=16).pack(side=tk.LEFT)
        self.cam_var = tk.StringVar(value="0")
        ttk.Entry(self.cam_row, textvariable=self.cam_var, width=5).pack(side=tk.LEFT, padx=4)

        # ---- 设置 ----
        set_frame = ttk.LabelFrame(main, text="参数设置", padding=8)
        set_frame.pack(fill=tk.X, pady=6)

        mrow = ttk.Frame(set_frame)
        mrow.pack(fill=tk.X)
        ttk.Label(mrow, text="比例尺:", width=8).pack(side=tk.LEFT)
        self.scale_mode = tk.StringVar(value="auto")
        ttk.Radiobutton(mrow, text="自动(按身高估算)", variable=self.scale_mode,
                        value="auto", command=self._toggle_scale).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Radiobutton(mrow, text="手动 m/px:", variable=self.scale_mode,
                        value="manual", command=self._toggle_scale).pack(side=tk.LEFT)
        self.scale_var = tk.StringVar(value="0.02")
        self.scale_entry = ttk.Entry(mrow, textvariable=self.scale_var, width=8, state="disabled")
        self.scale_entry.pack(side=tk.LEFT, padx=4)

        # ---- 按钮 ----
        btn_row = ttk.Frame(main)
        btn_row.pack(fill=tk.X, pady=8)
        self.start_btn = ttk.Button(btn_row, text="开始分析", command=self._start)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.stop_btn = ttk.Button(btn_row, text="停止", command=self._stop, state="disabled")
        self.stop_btn.pack(side=tk.LEFT)

        # ---- 状态 ----
        self.status_var = tk.StringVar(value="就绪 - 选择视频或切换摄像头模式")
        ttk.Label(main, textvariable=self.status_var, foreground="gray").pack(anchor=tk.W, pady=(8, 0))

        # 提示
        tip = ttk.LabelFrame(main, text="提示", padding=6)
        tip.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(tip, text="视频模式：选择 .mp4 文件\n摄像头模式：对着摄像头走几步\n分析结束后自动弹出速度报告和曲线图",
                  foreground="#555").pack(anchor=tk.W)

    def _toggle_source(self):
        mode = self.src_var.get()
        if mode == "file":
            self.cam_row.pack_forget()
            self.file_row.pack(fill=tk.X, pady=(6, 0))
            self.browse_btn.pack(side=tk.LEFT)
        else:
            self.file_row.pack_forget()
            self.browse_btn.pack_forget()
            self.cam_row.pack(fill=tk.X, pady=(6, 0))

    def _toggle_scale(self):
        state = "normal" if self.scale_mode.get() == "manual" else "disabled"
        self.scale_entry.config(state=state)

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("视频", "*.mp4 *.avi *.mov *.mkv"), ("所有", "*.*")]
        )
        if path:
            self.video_var.set(path)

    def _start(self):
        if self.src_var.get() == "file":
            path = self.video_var.get().strip()
            if not path or not os.path.isfile(path):
                messagebox.showwarning("提示", "请选择有效视频文件")
                return
            self.source_type = "file"
        else:
            try:
                cam_id = int(self.cam_var.get())
            except ValueError:
                cam_id = 0
            path = cam_id
            self.source_type = "camera"

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.running = True
        self.status_var.set("分析中... 按 Q 或 ESC 停止")

        self.thread = threading.Thread(target=self._run, args=(path,), daemon=True)
        self.thread.start()

    def _run(self, source):
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self._on_done("无法打开视频源")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30 if self.source_type == "file" else 15

            scale_mode = self.scale_mode.get()
            manual_scale = 0.02
            if scale_mode == "manual":
                try:
                    manual_scale = float(self.scale_var.get())
                except ValueError:
                    manual_scale = 0.02

            self.tracker = PersonTracker(fps=fps, scale_mode=scale_mode,
                                         manual_scale=manual_scale)

            window_name = "行人追踪 — 按Q/ESC退出"
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.tracker.process_frame(frame)
                cv2.imshow(window_name, result)

                wait_ms = max(1, int(1000 / fps)) if self.source_type == "file" else 1
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()

            self.root.after(0, self._show_report)
            self._on_done("分析完成！")
        except Exception as e:
            self._on_done(f"错误: {e}")

    def _stop(self):
        self.running = False
        self._on_done("已停止")

    def _on_done(self, msg):
        def _update():
            self.status_var.set(msg)
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
        self.root.after(0, _update)

    def _show_report(self):
        """弹出报告窗口：控制台表格 + matplotlib 曲线图"""
        if not self.tracker:
            return

        report = self.tracker.get_report_data()
        if not report:
            self.status_var.set("分析完成 — 未检测到有效行人数据")
            return

        # 打印控制台报告
        print("\n" + "=" * 55)
        print("           行人速率分析报告")
        print("=" * 55)
        print(f"{'ID':<6} {'平均速度(m/s)':<16} {'帧数':<8} {'时长(s)':<10}")
        print("-" * 55)
        for p in report:
            print(f"{p['id']:<6} {p['avg_speed']:<16.3f} {p['total_frames']:<8} {p['time_seconds']:<10.2f}")
        overall = sum(p["avg_speed"] for p in report) / len(report)
        print("=" * 55)
        print(f"  {len(report)} 人 | 平均: {overall:.3f} m/s")
        print("=" * 55 + "\n")

        # 曲线图独立窗口
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.tab10.colors
        for i, p in enumerate(report):
            t = [f / self.tracker.fps for f in p["frames"]]
            ax.plot(t, p["speeds"], linewidth=1.5, color=colors[i % 10],
                    label=f"ID:{p['id']} ({p['avg_speed']:.2f} m/s)")

        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Speed (m/s)", fontsize=12)
        ax.set_title("Speed Change Over Time", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        plt.show()

        self.status_var.set(f"报告完成 — {len(report)} 人追踪，见控制台和图")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    root = tk.Tk()
    app = SpeedApp(root, video_path=video_path)
    root.mainloop()
