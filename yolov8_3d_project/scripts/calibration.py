# scripts/calibration.py
import cv2
import numpy as np
import glob
import yaml
import os
import argparse

def calibrate_camera(images_path, chessboard_size, square_size, output_file):
    # Persiapan titik objek (3D point di dunia nyata)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[1], 0:chessboard_size[0]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # titik 3D di dunia nyata
    imgpoints = []  # titik 2D di citra

    images = glob.glob(os.path.join(images_path, "*.jpg"))
    if len(images) == 0:
        print(f"[ERROR] Tidak ada gambar ditemukan di {images_path}")
        return

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Cari pola chessboard
        ret, corners = cv2.findChessboardCorners(
            gray, (chessboard_size[1], chessboard_size[0]), None
        )

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            imgpoints.append(corners2)

            # Visualisasi (opsional)
            cv2.drawChessboardCorners(img, (chessboard_size[1], chessboard_size[0]), corners2, ret)
            cv2.imshow("Calibration", img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    # Kalibrasi kamera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("[INFO] Matriks Kamera:\n", mtx)
    print("[INFO] Koefisien Distorsi:\n", dist)

    # Simpan ke YAML
    data = {
        "camera": {
            "fx": float(mtx[0, 0]),
            "fy": float(mtx[1, 1]),
            "cx": float(mtx[0, 2]),
            "cy": float(mtx[1, 2]),
            "k1": float(dist[0, 0]),
            "k2": float(dist[0, 1]),
            "p1": float(dist[0, 2]),
            "p2": float(dist[0, 3]),
            "k3": float(dist[0, 4]) if len(dist[0]) > 4 else 0.0,
        }
    }

    with open(output_file, "w") as f:
        yaml.dump(data, f)

    print(f"[INFO] Parameter kamera disimpan ke {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalibrasi Kamera")
    parser.add_argument("--config", default="config/params.yaml", help="Path file params.yaml")
    parser.add_argument("--cam", default="cam0", help="ID kamera (cam0, cam1, cam2)")
    args = parser.parse_args()

    # Load params.yaml
    with open(args.config, "r") as f:
        params = yaml.safe_load(f)

    chessboard_size = (
        params["calibration"]["chessboard_rows"],
        params["calibration"]["chessboard_cols"],
    )
    square_size = params["calibration"]["square_size"]
    images_path = params["calibration"]["images_path"]

    output_file = os.path.join("config", f"{args.cam}.yaml")

    calibrate_camera(images_path, chessboard_size, square_size, output_file)
