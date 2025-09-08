import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO

# ==============================
# KONFIGURASI
# ==============================
cam_urls = [
    "http://10.57.172.211:8080/video",  # cam0
    "http://10.57.172.228:8080/video",  # cam1
    "http://10.57.172.87:8080/video"    # cam2
]

class_colors = {
    "ikan_mas": (0, 255, 255),     # Kuning
    "ikan_mujair": (128, 128, 128) # Abu-abu
}

# Path model YOLOv8 (pastikan sudah ada file best.pt di folder)
model_path = "best.pt"
model = YOLO(model_path)

# ==============================
# KALMAN FILTER 3D
# ==============================
def create_kalman():
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.F = np.array([[1,0,0,1,0,0],
                     [0,1,0,0,1,0],
                     [0,0,1,0,0,1],
                     [0,0,0,1,0,0],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,1]])
    kf.H = np.array([[1,0,0,0,0,0],
                     [0,1,0,0,0,0],
                     [0,0,1,0,0,0]])
    kf.P *= 1000
    kf.R = np.eye(3) * 0.01
    kf.Q = np.eye(6) * 0.01
    return kf

kalman_filters = {}

# ==============================
# CAMERA
# ==============================
caps = [cv2.VideoCapture(url) for url in cam_urls]

# ==============================
# KONFIGURASI TRIANGULASI DUMMY
# ==============================
def triangulate_3cam(pt1, pt2, pt3):
    x = (pt1[0] + pt2[0] + pt3[0]) / 3
    y = (pt1[1] + pt2[1] + pt3[1]) / 3
    z = 1000 / (abs(pt1[0]-pt2[0])+abs(pt2[0]-pt3[0])+1)  # perkiraan jarak sederhana
    return np.array([x, y, z])

# ==============================
# LOOP UTAMA
# ==============================
while True:
    frames = []
    for idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] cam{idx} tidak dapat frame")
            frame = np.zeros((480,640,3), dtype=np.uint8)
        frames.append(frame)

    # DETEKSI YOLOv8
    detections_per_cam = []
    for frame in frames:
        results = model(frame)
        detections_per_cam.append(results.pandas().xyxy[0])

    # TRIANGULASI & KALMAN
    objects_3d = []
    for cls_name in ["ikan_mas", "ikan_mujair"]:
        pts = []
        for det in detections_per_cam:
            df_cls = det[det['name']==cls_name]
            if len(df_cls)>0:
                x = (df_cls['xmin'].values[0] + df_cls['xmax'].values[0])/2
                y = (df_cls['ymin'].values[0] + df_cls['ymax'].values[0])/2
                pts.append((x,y))
        if len(pts)==3:
            pos_3d = triangulate_3cam(pts[0], pts[1], pts[2])
            if cls_name not in kalman_filters:
                kalman_filters[cls_name] = create_kalman()
            kf = kalman_filters[cls_name]
            kf.predict()
            kf.update(pos_3d)
            pos_3d = kf.x[:3]
            objects_3d.append((cls_name, pos_3d))

    # GAMBAR BOUNDING BOX 3D
    for cls_name, pos_3d in objects_3d:
        x, y, z = pos_3d
        size = int(max(20, 200/z))
        cv2.rectangle(frames[0],
                      (int(x-size/2), int(y-size/2)),
                      (int(x+size/2), int(y+size/2)),
                      class_colors[cls_name], 2)
        cv2.putText(frames[0], cls_name, (int(x-size/2), int(y-size/2)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors[cls_name], 2)

    # TAMPILKAN SEMUA CAM DIGABUNG
    combined_frame = np.hstack(frames)
    cv2.imshow("YOLOv8 3D Bounding Box", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# RELEASE
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
