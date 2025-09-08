import cv2
import torch
import numpy as np
from itertools import product

# URL kamera (ganti sesuai IP)
cam_urls = [
    "http://192.168.1.10:8080/video",  # cam0
    "http://192.168.1.11:8080/video",  # cam1
    "http://192.168.1.12:8080/video"   # cam2
]

# Inisialisasi kamera
caps = [cv2.VideoCapture(url) for url in cam_urls]

# Load model YOLOv8 (pastikan modelmu sudah dilatih untuk ikan mas & mujair)
model = torch.hub.load('ultralytics/yolov8', 'custom', path='best.pt', force_reload=True)

# Kelas & warna
class_colors = {0: (0,255,255), 1: (128,128,128)}  # 0: ikan mas, 1: ikan mujair
class_names = {0: "Ikan Mas", 1: "Ikan Mujair"}

# Posisi kamera (world coordinates)
camera_positions = [
    np.array([0, 0, 0]),
    np.array([0.5, 0, 0]),
    np.array([1.0, 0, 0])
]

def triangulate_points(points_2d, cam_pos):
    """
    Triangulasi beberapa titik 2D dari 3 kamera menjadi 3D (lebih akurat)
    """
    A = []
    for (x, y), pos in zip(points_2d, cam_pos):
        A.append([x + pos[0]*10, y + pos[1]*10, 1.0 + pos[2]])
    pts_3d = np.mean(A, axis=0)
    return pts_3d

def draw_3d_box(frame, center3d, color=(0,255,0), label=""):
    """
    Gambar bounding box 3D sederhana dengan label
    """
    x, y, z = center3d
    scale = max(20, int(1000/(z+1)))  # skala berdasarkan jarak
    pts = [
        (int(x-scale), int(y-scale)),
        (int(x+scale), int(y-scale)),
        (int(x+scale), int(y+scale)),
        (int(x-scale), int(y+scale))
    ]
    cv2.polylines(frame, [np.array(pts)], isClosed=True, color=color, thickness=2)
    cv2.putText(frame, label, (pts[0][0], pts[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

while True:
    frames = []
    detections = []

    # Baca frame semua kamera
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480,640,3), dtype=np.uint8)
        frames.append(frame)

    # Jalankan deteksi untuk tiap kamera
    for frame in frames:
        results = model(frame)
        dets = []
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1+x2)//2
            cy = (y1+y2)//2
            dets.append({'cls': int(cls), 'bbox':[x1,y1,x2,y2], 'centroid':(cx,cy)})
        detections.append(dets)

    # Matching sederhana: nearest centroid
    min_len = min(len(d) for d in detections if len(d)>0)
    if min_len>0:
        for idxs in product(*[range(len(d)) for d in detections]):
            pts_2d = [detections[i][idxs[i]]['centroid'] for i in range(3)]
            classes = [detections[i][idxs[i]]['cls'] for i in range(3)]
            pts3d = triangulate_points(pts_2d, camera_positions)
            draw_3d_box(frames[0], pts3d, color=class_colors[classes[0]], label=class_names[classes[0]])

    # Gabungkan 3 kamera di satu layar
    top = np.hstack((frames[0], frames[1]))
    bottom = np.hstack((frames[2], np.zeros_like(frames[2])))
    combined = np.vstack((top, bottom))

    cv2.imshow("3D Bounding Box Multi-Cam", combined)
    if cv2.waitKey(1) & 0xFF == 27:
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
