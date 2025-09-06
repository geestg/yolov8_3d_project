import os

# Folder utama
base_dir = "yolov8_3d_project"

# Struktur folder
folders = [
    "data/images/train",
    "data/images/val",
    "data/labels/train",
    "data/labels/val",
    "calibration/checkerboard_images",
    "calibration/left",
    "calibration/right",
    "models",
    "scripts",
    "config",
    "results/runs",
    "results/output_videos"
]

# Buat folder satu per satu
for folder in folders:
    path = os.path.join(base_dir, folder)
    os.makedirs(path, exist_ok=True)

# Buat file kosong penting
files = {
    "calibration/calibrate.py": "# Script kalibrasi stereo vision\n",
    "scripts/train_yolo.py": "# Script training YOLOv8\n",
    "scripts/test_yolo.py": "# Script testing YOLOv8\n",
    "scripts/stereo_test.py": "# Script uji stereo depth\n",
    "scripts/yolo_stereo.py": "# Integrasi YOLOv8 + stereo vision\n",
    "config/dataset.yaml": "path: ../data\ntrain: images/train\nval: images/val\n\nnames:\n  0: ikan_mas\n  1: ikan_mujair\n",
    "config/params.yaml": "# Parameter kalibrasi stereo\n",
    "requirements.txt": "ultralytics\nopencv-python\nnumpy\nmatplotlib\n"
}

for file_path, content in files.items():
    path = os.path.join(base_dir, file_path)
    with open(path, "w") as f:
        f.write(content)

print("✅ Struktur folder YOLOv8 3D berhasil dibuat!")
