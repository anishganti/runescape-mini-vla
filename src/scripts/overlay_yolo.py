import time
import os
import pywinctl as pwc
import mss
import numpy as np
import cv2
from ultralytics import YOLO

# -------------------------------
WINDOW_TITLE = "Old School RuneScape"
MODEL_PATH = "./runs/detect/runescape_ore/weights/best.pt"
CONF_THRESH = 0.25
SAVE_DIR = "./data/live_capture"
IMG_DIR = os.path.join(SAVE_DIR, "images")
LABEL_DIR = os.path.join(SAVE_DIR, "labels")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# -------------------------------
def get_window():
    wins = pwc.getWindowsWithTitle(WINDOW_TITLE)
    if not wins:
        raise RuntimeError(f"Window '{WINDOW_TITLE}' not found")
    win = wins[0]
    win.activate()
    time.sleep(0.2)
    return win

def get_window_info(win):
    return {"left": win.left, "top": win.top, "width": win.width, "height": win.height}

def convert_bgra_to_rgb(frame):
    return frame[:, :, :3][:, :, ::-1]

def save_yolo_labels(boxes, img_path, img_shape):
    h, w = img_shape[:2]
    lines = []
    for (x1, y1, x2, y2) in boxes:
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"0 {xc} {yc} {bw} {bh}")
    label_path = os.path.join(LABEL_DIR, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
    with open(label_path, "w") as f:
        f.write("\n".join(lines))

# -------------------------------
model = YOLO(MODEL_PATH)
# -------------------------------
def main():
    win = get_window()
    bbox = get_window_info(win)
    sct = mss.mss()

    cv2.namedWindow("RuneScape + YOLO", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RuneScape + YOLO", bbox["width"]*2, bbox["height"])

    frame_id = 0
    print("Press S to save YOLO frame, ESC to quit.")

    while True:
        img = sct.grab(bbox)
        frame = convert_bgra_to_rgb(np.array(img))
        frame = np.ascontiguousarray(frame)

        # Copy for YOLO predictions
        yolo_frame = frame.copy()
        boxes_to_save = []

        # Run YOLO
        results = model.predict(frame, conf=CONF_THRESH, verbose=False)
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), conf in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(yolo_frame, f"{conf:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                boxes_to_save.append((x1, y1, x2, y2))

        # Concatenate original + YOLO frame side by side
        disp_frame = np.concatenate((frame, yolo_frame), axis=1)
        disp_frame = cv2.cvtColor(disp_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("RuneScape + YOLO", disp_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            # Save the YOLO frame + labels
            img_path = os.path.join(IMG_DIR, f"frame_{frame_id:06d}.png")
            cv2.imwrite(img_path, cv2.cvtColor(yolo_frame, cv2.COLOR_RGB2BGR))
            save_yolo_labels(boxes_to_save, img_path, yolo_frame.shape)
            print(f"Saved {img_path} with {len(boxes_to_save)} boxes")
            frame_id += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
