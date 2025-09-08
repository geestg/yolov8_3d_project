import os

# Daftar folder yang perlu dibuat
folders = [
    "config",
    "datasets/calib/cam0",
    "datasets/calib/cam1",
    "datasets/calib/cam2",
    "datasets/calib/pairs_cam0_cam1",
    "datasets/calib/pairs_cam0_cam2",
    "datasets/images/train",
    "datasets/images/val",
    "datasets/labels/train",
    "datasets/labels/val",
    "scripts",
    "models",
    "results/logs",
    "results/disparity",
    "results/outputs",
]

# File kosong (placeholder) yang akan dibuat
files = [
    "config/params.yaml",
    "config/cam0.yaml",
    "config/cam1.yaml",
    "config/cam2.yaml",
    "config/stereo_cam0_cam1.yaml",
    "config/stereo_cam0_cam2.yaml",
    "datasets/data_fish.yaml",
    "scripts/calibration.py",
    "scripts/stereo_calibrate.py",
    "scripts/utils_3d.py",
    "scripts/yolo_stereo.py",
]

def create_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"[OK] Folder dibuat: {folder}")

    for file in files:
        if not os.path.exists(file):
            with open(file, "w") as f:
                f.write("")  # bikin file kosong
            print(f"[OK] File dibuat: {file}")
        else:
            print(f"[SKIP] File sudah ada: {file}")

if __name__ == "__main__":
    create_structure()
    print("\nStruktur proyek YOLOv8 3D siap ✅")
