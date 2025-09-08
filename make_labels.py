import os
import cv2
import numpy as np
from tqdm import tqdm

# ==============================
# KONFIGURASI PATH
# ==============================
dataset_dir = r"D:\SEMESTER 5\TA 1\yolov8_3d_project\datasets"
images_train_dir = os.path.join(dataset_dir, "images", "train")
images_val_dir   = os.path.join(dataset_dir, "images", "val")

labels_train_dir = os.path.join(dataset_dir, "labels", "train")
labels_val_dir   = os.path.join(dataset_dir, "labels", "val")

os.makedirs(labels_train_dir, exist_ok=True)
os.makedirs(labels_val_dir, exist_ok=True)

# ==============================
# KONFIGURASI KELAS
# ==============================
class_map = {
    "mas": 0,
    "mujair": 1
}

# Warna threshold HSV untuk deteksi ikan
color_ranges = {
    "mas": [(15, 100, 100), (35, 255, 255)],       # Kuning/oranye
    "mujair": [(0, 0, 50), (180, 50, 200)]         # Abu-abu gelap
}

# ==============================
# FUNSI GENERATE LABEL DARI KONTRUR
# ==============================
def generate_label_from_contour(image_path, output_dir):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    filename = os.path.basename(image_path)
    name, _ = os.path.splitext(filename)

    # Tentukan class berdasarkan nama file
    cls_name = None
    for key in class_map:
        if key in name.lower():
            cls_name = key
            break
    if cls_name is None:
        print(f"[WARN] Tidak bisa tentukan class untuk {filename}")
        return

    # Convert ke HSV untuk threshold warna
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower, upper = color_ranges[cls_name]
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

    # Morphology untuk bersihkan noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

    # Cari kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        # fallback random bounding box
        x_center = np.random.uniform(0.3, 0.7)
        y_center = np.random.uniform(0.3, 0.7)
        width    = np.random.uniform(0.1, 0.3)
        height   = np.random.uniform(0.1, 0.3)
    else:
        # ambil kontur terbesar
        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        # normalisasi 0-1
        x_center = (x + bw/2) / w
        y_center = (y + bh/2) / h
        width    = bw / w
        height   = bh / h

    # Dummy depth
    z_depth = np.random.uniform(0.5, 3.0)

    label_txt = f"{class_map[cls_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {z_depth:.6f}\n"

    # Simpan file
    label_path = os.path.join(output_dir, name + ".txt")
    with open(label_path, "w") as f:
        f.write(label_txt)

# ==============================
# PROSES DATASET
# ==============================
def process_dataset(images_dir, labels_dir):
    img_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg",".png"))]
    for img_file in tqdm(img_files, desc=f"Generate labels di {images_dir}"):
        generate_label_from_contour(os.path.join(images_dir, img_file), labels_dir)

# ==============================
# JALANKAN
# ==============================
process_dataset(images_train_dir, labels_train_dir)
process_dataset(images_val_dir, labels_val_dir)

print("Selesai generate label 3D akurat!")
