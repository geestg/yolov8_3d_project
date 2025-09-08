import cv2
import os
import threading
import time
import numpy as np
from collections import deque

class CameraThread:
    def __init__(self, url, name="cam"):
        self.url = url
        self.name = name
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        self.thread.start()

    def update(self):
        while self.running:
            for _ in range(3):  # buang buffer biar lebih realtime
                ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

                # Hitung FPS sederhana
                self.frame_count += 1
                if self.frame_count >= 10:
                    now = time.time()
                    dt = now - self.last_time
                    if dt > 0:
                        self.fps = self.frame_count / dt
                    self.frame_count = 0
                    self.last_time = now
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None, self.fps

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


def main():
    save_dir = "datasets/calib/triplets_cam0_cam1_cam2"
    os.makedirs(save_dir, exist_ok=True)
    idx = len(os.listdir(save_dir)) // 3

    cam0 = CameraThread("http://10.57.172.211:8080/video", "cam0")
    cam1 = CameraThread("http://10.57.172.228:8080/video", "cam1")
    cam2 = CameraThread("http://10.57.172.87:8080/video", "cam2")

    time.sleep(2)  # tunggu stream stabil

    while True:
        f0, fps0 = cam0.read()
        f1, fps1 = cam1.read()
        f2, fps2 = cam2.read()

        if f0 is None or f1 is None or f2 is None:
            print("⚠️ Menunggu semua stream siap...")
            time.sleep(0.1)
            continue

        # Simpan raw frame (full resolusi) untuk dataset
        raw0, _ = cam0.read()
        raw1, _ = cam1.read()
        raw2, _ = cam2.read()

        # Resize untuk preview supaya ringan
        disp0 = cv2.resize(f0, (426, 320))
        disp1 = cv2.resize(f1, (426, 320))
        disp2 = cv2.resize(f2, (426, 320))

        # Tambah label FPS
        cv2.putText(disp0, f"cam0 | {fps0:.1f} FPS", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(disp1, f"cam1 | {fps1:.1f} FPS", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(disp2, f"cam2 | {fps2:.1f} FPS", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        empty = np.zeros_like(disp0)

        # Layout grid
        row1 = np.hstack([disp0, disp1])
        row2 = np.hstack([disp2, empty])
        preview = np.vstack([row1, row2])

        # Instruksi di bawah
        cv2.putText(preview, "Press 's'=save triplet   'q'=quit",
                    (10, preview.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        cv2.imshow("Stereo Preview [cam0 | cam1 | cam2]", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if raw0 is not None and raw1 is not None and raw2 is not None:
                cv2.imwrite(os.path.join(save_dir, f"triplet_{idx:03d}_cam0.jpg"), raw0)
                cv2.imwrite(os.path.join(save_dir, f"triplet_{idx:03d}_cam1.jpg"), raw1)
                cv2.imwrite(os.path.join(save_dir, f"triplet_{idx:03d}_cam2.jpg"), raw2)
                print(f"✅ Disimpan triplet {idx}")
                idx += 1
        elif key == ord('q'):
            break

    cam0.stop()
    cam1.stop()
    cam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
