import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time

# =====================
# CAMERA THREAD
# =====================
class CameraThread(threading.Thread):
    def __init__(self, url, name="Camera"):
        super().__init__()
        self.url = url
        self.name = name
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                else:
                    print(f"[WARN] {self.name} tidak dapat frame")
                    time.sleep(0.1)
            except cv2.error as e:
                print(f"[ERROR] {self.name} OpenCV: {e}")
                time.sleep(1)

    def stop(self):
        self.running = False
        self.cap.release()

# =====================
# CAMERA URL
# =====================
cam0 = CameraThread("http://10.57.172.211:8080/video", "cam0")
cam1 = CameraThread("http://10.57.172.228:8080/video", "cam1")
cam2 = CameraThread("http://10.57.172.87:8080/video", "cam2")
cams = [cam0, cam1, cam2]

# =====================
# YOLO MODEL
# =====================
MODEL_PATH = "yolov8n.pt"  # ganti jika pakai model ikan
CLASS_NAMES = ["person", "bicycle", "car", "motorbike"]  # COCO default (untuk debug)
model = YOLO(MODEL_PATH)

# =====================
# DRAW 3D BBOX
# =====================
def draw_3d_bbox(img, center, size=(5,2,2), color=(0,255,0)):
    x, y, z = center
    l, w, h = size
    corners = np.array([
        [x-l/2, y-w/2, z-h/2],
        [x+l/2, y-w/2, z-h/2],
        [x+l/2, y+w/2, z-h/2],
        [x-l/2, y+w/2, z-h/2],
        [x-l/2, y-w/2, z+h/2],
        [x+l/2, y-w/2, z+h/2],
        [x+l/2, y+w/2, z+h/2],
        [x-l/2, y+w/2, z+h/2],
    ])
    corners_2d = (corners[:, :2]*10 + 200).astype(int)
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for e in edges:
        cv2.line(img, tuple(corners_2d[e[0]]), tuple(corners_2d[e[1]]), color, 2)

# =====================
# PIPELINE DETEKSI 3 CAM
# =====================
def run_pipeline():
    # Start all camera threads
    for cam in cams:
        cam.start()

    cv2.namedWindow("YOLO 3D 3Cam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO 3D 3Cam", 1280, 720)

    try:
        while True:
            frames = [cam.frame for cam in cams]
            if any(f is None for f in frames):
                time.sleep(0.05)
                continue

            # Resize frame agar bisa digabung
            resized_frames = [cv2.resize(f, (320,240)) for f in frames]

            # Deteksi YOLO untuk setiap frame
            for i in range(3):
                img = resized_frames[i]
                results = model.predict(img, conf=0.3)
                for r in results:
                    for det in r.boxes.data.tolist():
                        x1, y1, x2, y2, score, cls = det
                        cls = int(cls)
                        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else "unknown"
                        length_cm = round((x2-x1)/10,1)
                        center2d = ((x1+x2)/2, (y1+y2)/2)
                        # Draw 2D rectangle
                        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                        cv2.putText(img, f"{label}:{length_cm}cm", (int(x1),int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
                        # Draw 3D cube
                        draw_3d_bbox(img, (center2d[0], center2d[1], 5), size=(length_cm, length_cm/2, length_cm/2))

            # Gabungkan 3 frame horizontal
            combined = cv2.hconcat(resized_frames)
            cv2.imshow("YOLO 3D 3Cam", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        for cam in cams:
            cam.stop()
        cv2.destroyAllWindows()

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    run_pipeline()
