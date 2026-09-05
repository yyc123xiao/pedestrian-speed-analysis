"""从摄像头录制 10 秒测试视频"""
import cv2

DURATION = 10
OUTPUT = "test_video.mp4"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("摄像头未找到！请确认摄像头已连接。")
    exit(1)

fps = 20
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT, fourcc, fps, (w, h))

print(f"录制 {DURATION} 秒测试视频... 请在摄像头前走动。")
print("按 Q 提前结束。")

frame_count = 0
while frame_count < fps * DURATION:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"Recording... {frame_count}/{fps*DURATION}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Recording - 录制中", frame)
    out.write(frame)
    frame_count += 1

    if cv2.waitKey(50) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"测试视频已保存: {OUTPUT} ({frame_count} 帧)")
