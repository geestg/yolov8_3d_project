#!/usr/bin/env python3
"""
YOLOv8 + Stereo Depth (robust version)
- fallback untuk 1 kamera (simulasi)
- membaca params.yaml (kalibrasi) dengan fallback default
- pilihan sumber kamera via argumen (index atau URL)
- memakai CAP_DSHOW di Windows untuk stabilitas
"""

import cv2
import numpy as np
import yaml
import argparse
import time
from ultralytics import YOLO
from pathlib import Path

# -------------------------
# Argparse: sumber kamera
# -------------------------
parser = argparse.ArgumentParser(description="YOLOv8 + Stereo Depth (robust)")
parser.add_argument("--left", type=str, default="0",
                    help="Left camera source (index or URL). Default=0")
parser.add_argument("--right", type=str, default=None,
                    help="Right camera source (index or URL). If not set, will try next index or duplicate left.")
parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model path")
parser.add_argument("--width", type=int, default=640, help="Frame width")
parser.add_argument("--height", type=int, default=480, help="Frame height")
parser.add_argument("--use_dshow", action="store_true", help="Use CAP_DSHOW backend on Windows")
args = parser.parse_args()

# -------------------------
# Helper: parse source into cv2.VideoCapture arguments
# -------------------------
def open_capture(source, use_dshow=False, width=640, height=480):
    """
    source: string. If numeric string -> treat as index int; else treat as URL/file.
    """
    try:
        if source is None:
            return None
        if source.isdigit():
            idx = int(source)
            backend = cv2.CAP_DSHOW if use_dshow else cv2.CAP_ANY
            cap = cv2.VideoCapture(idx, backend)
        else:
            # URL or filename
            cap = cv2.VideoCapture(source)
        # set resolution (best effort)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        time.sleep(0.2)
        return cap
    except Exception as e:
        print(f"❌ Error open_capture({source}): {e}")
        return None

# -------------------------
# Load calibration params.yaml (with fallback)
# -------------------------
params_path = Path("config/params.yaml")
focal_length = 700.0
baseline = 0.1
if params_path.exists():
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        if params is None:
            params = {}
        focal_length = float(params.get("focal_length", focal_length))
        baseline = float(params.get("baseline", baseline))
    except Exception as e:
        print(f"⚠️ Gagal baca config/params.yaml, pakai fallback. Error: {e}")
else:
    print("⚠️ config/params.yaml tidak ditemukan — memakai nilai default (focal_length=700, baseline=0.1 m)")

print(f"✅ Loaded calibration: Focal length = {focal_length}, Baseline = {baseline} m")

# -------------------------
# Load YOLOv8 model
# -------------------------
try:
    model = YOLO(args.model)
    print(f"✅ Loaded YOLO model: {args.model}")
except Exception as e:
    print(f"❌ Gagal load model '{args.model}': {e}")
    print("Pastikan model ada atau koneksi internet untuk mendownload model default.")
    raise

# -------------------------
# Buka capture left & right
# -------------------------
use_dshow = args.use_dshow
cap_left = open_capture(args.left, use_dshow=use_dshow, width=args.width, height=args.height)

# Determine right source: explicit, or try next index, or duplicate left
cap_right = None
if args.right is not None:
    cap_right = open_capture(args.right, use_dshow=use_dshow, width=args.width, height=args.height)
else:
    # if left is numeric index, try left+1
    if args.left.isdigit():
        try_idx = str(int(args.left) + 1)
        cap_right = open_capture(try_idx, use_dshow=use_dshow, width=args.width, height=args.height)
    # if still None, duplicate left (simulation)
    if cap_right is None:
        print("⚠️ Kamera kanan tidak tersedia: akan menggunakan simulasi (duplikasi) dari kamera kiri.")
        # we will duplicate frames from left later

if cap_left is None or (cap_right is None and not args.right and not args.left.isdigit()):
    # If left missing completely => error
    if cap_left is None:
        print("❌ Tidak ada kamera kiri yang bisa dibuka. Periksa index atau URL.")
        raise SystemExit(1)

# -------------------------
# Stereo matcher (SGBM) setup
# -------------------------
window_size = 5
min_disp = 0
num_disp = 16 * 6  # must be divisible by 16

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=11,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# -------------------------
# Utility: normalize disparity for display
# -------------------------
def normalize_disparity(disp, min_disp=min_disp, num_disp=num_disp):
    """
    Normalize disparity (float32) to 0..255 uint8
    """
    disp_vis = np.copy(disp)
    # mask invalid values
    disp_vis[disp_vis <= 0] = min_disp
    disp_vis = (disp_vis - min_disp) / float(num_disp)
    disp_vis = np.clip(disp_vis, 0.0, 1.0)
    disp_vis = (disp_vis * 255).astype(np.uint8)
    return disp_vis

# -------------------------
# Main loop
# -------------------------
print("▶️ Menjalankan loop utama. Tekan 'q' untuk keluar.")
while True:
    # baca frame left
    retL, frame_left = cap_left.read()
    if not retL or frame_left is None:
        # jangan exit langsung; coba lagi
        print("⚠️ Gagal membaca frame kiri — mencoba ulang...")
        time.sleep(0.1)
        continue

    # baca frame right (jika ada capture object)
    if cap_right is not None:
        retR, frame_right = cap_right.read()
        if not retR or frame_right is None:
            # jika gagal baca kanan, gunakan duplikasi (simulasi)
            print("⚠️ Gagal membaca frame kanan — pakai duplikasi/geser (simulasi).")
            frame_right = frame_left.copy()
            # optional: create slight shift to help disparity algorithm
            shift = 2
            frame_right = np.roll(frame_right, shift, axis=1)
    else:
        # no right capture => simulate right by shifting
        frame_right = frame_left.copy()
        shift = 2
        frame_right = np.roll(frame_right, shift, axis=1)

    # pastikan ukuran sama
    frame_left = cv2.resize(frame_left, (args.width, args.height))
    frame_right = cv2.resize(frame_right, (args.width, args.height))

    # grayscale untuk stereo
    grayL = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # compute disparity (SGBM)
    try:
        disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    except Exception as e:
        print(f"⚠️ Error compute disparity: {e}")
        # buat array nol sebagai fallback
        disparity = np.zeros_like(grayL, dtype=np.float32)

    # jalankan YOLO di frame kiri
    try:
        results = model(frame_left, verbose=False)  # returns list-like object
        res0 = results[0]  # first result
    except Exception as e:
        print(f"⚠️ Error running YOLO: {e}")
        res0 = None

    display_frame = frame_left.copy()

    if res0 is not None and hasattr(res0, "boxes"):
        boxes = res0.boxes.xyxy.cpu().numpy() if len(res0.boxes) > 0 else np.array([])
        classes = res0.boxes.cls.cpu().numpy() if len(res0.boxes) > 0 else np.array([])
        confs = res0.boxes.conf.cpu().numpy() if len(res0.boxes) > 0 else np.array([])

        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            # clamp
            x1 = max(0, min(x1, args.width - 1))
            x2 = max(0, min(x2, args.width - 1))
            y1 = max(0, min(y1, args.height - 1))
            y2 = max(0, min(y2, args.height - 1))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # ambil beberapa nilai disparity dalam ROI untuk robust estimate
            roi = disparity[y1:y2, x1:x2]
            if roi.size == 0:
                disp_value = 0.0
            else:
                roi_valid = roi[roi > 0]
                if roi_valid.size == 0:
                    # fallback pakai median dari roi (may be zeros)
                    disp_value = float(np.median(roi))
                else:
                    disp_value = float(np.median(roi_valid))

            # perhitungan kedalaman Z (meter), cek div by zero
            if disp_value > 0:
                depth = (focal_length * baseline) / disp_value
            else:
                depth = float("nan")

            # gambar bounding box dan label
            cls_id = int(cls) if not np.isnan(cls) else -1
            label_text = f"Class {cls_id} | d={depth:.2f} m | conf={conf:.2f}"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(display_frame, label_text, (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    else:
        # jika tidak ada hasil YOLO, tampilkan teks
        cv2.putText(display_frame, "No YOLO result", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    # visualize disparity
    disp_vis = normalize_disparity(disparity, min_disp=min_disp, num_disp=num_disp)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # stack display
    try:
        stacked = np.hstack((cv2.resize(display_frame, (args.width, args.height)), disp_color))
    except Exception:
        stacked = display_frame

    cv2.imshow("YOLO + Depth (Left)", stacked)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# cleanup
if cap_left is not None:
    cap_left.release()
if cap_right is not None:
    cap_right.release()
cv2.destroyAllWindows()
