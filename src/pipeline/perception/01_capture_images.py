import os
import re
import time

import pywinctl as pwc
import mss
import numpy as np
from PIL import Image
from pynput import keyboard
import cv2

# -------------------------------
# Config
# -------------------------------
IMAGES_DIR = "/Users/anishganti/runescape_mini_vla/data/mines/lumbridge/images"
LABELS_DIR = "/Users/anishganti/runescape_mini_vla/data/mines/lumbridge/labels"

WINDOW_TITLE = "Old School RuneScape"
CLASS_ID = 0  # only one class: ore

# -------------------------------
# Globals
# -------------------------------
frame_id = 0
STOP = False
capture_flag = False

# -------------------------------
# Utilities
# -------------------------------
def get_frame_id(images_dir):
    """Get next frame_id from existing folder."""
    if not os.path.exists(images_dir):
        return 0

    ids = []
    for fname in os.listdir(images_dir):
        m = re.match(r"frame_(\d+)\.png", fname)
        if m:
            ids.append(int(m.group(1)))

    return max(ids) + 1 if ids else 0


def get_window():
    wins = pwc.getWindowsWithTitle(WINDOW_TITLE)
    if not wins:
        raise RuntimeError(f"Window '{WINDOW_TITLE}' not found")
    return wins[0]


def get_window_info(win):
    return {"left": win.left, "top": win.top, "width": win.width, "height": win.height}


def convert_bgra_to_rgb(frame):
    return frame[:, :, :3][:, :, ::-1]

def convert_bgra_to_bgr(frame):
    return frame[:, :, :3]


def capture_frame(sct, bbox):
    img = sct.grab(bbox)
    rgb_frame = convert_bgra_to_rgb(np.array(img))
    bgr_frame = convert_bgra_to_bgr(np.array(img))
    return rgb_frame, bgr_frame


def save_frame(frame, frame_id):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    path = os.path.join(IMAGES_DIR, f"frame_{frame_id:06d}.png")
    Image.fromarray(frame).save(path)


def save_labels(labels, frame_id):
    os.makedirs(LABELS_DIR, exist_ok=True)
    path = os.path.join(LABELS_DIR, f"frame_{frame_id:06d}.txt")
    with open(path, "w") as f:
        for label in labels:
            f.write(" ".join(map(str, label)) + "\n")


# -------------------------------
# Annotation
# -------------------------------
def annotate_frame(frame):
    """
    Manual bounding box annotation using OpenCV.
    Controls:
      - Left click + drag: draw box
      - Enter: accept frame
      - r: reset boxes
      - Esc: discard frame
    Returns:
        List of YOLO-format labels
    """
    clone = frame.copy()
    h, w, _ = frame.shape
    boxes = []
    drawing = False
    ix, iy = -1, -1

    def mouse_callback(event, x, y, flags, param):
        nonlocal ix, iy, drawing, clone, boxes

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            temp = clone.copy()
            cv2.rectangle(temp, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Annotate", temp)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(clone, (ix, iy), (x, y), (0, 255, 0), 2)
            boxes.append((ix, iy, x, y))
            cv2.imshow("Annotate", clone)

    cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotate", mouse_callback)
    cv2.imshow("Annotate", clone)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            break
        elif key == ord("r"):  # Reset
            clone = frame.copy()
            boxes = []
            cv2.imshow("Annotate", clone)
        elif key == 27:  # Esc
            boxes = []
            break

    cv2.destroyAllWindows()

    # Convert to YOLO format
    yolo_labels = []
    for x1, y1, x2, y2 in boxes:
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        bw = abs(x2 - x1) / w
        bh = abs(y2 - y1) / h
        yolo_labels.append([CLASS_ID, x_center, y_center, bw, bh])

    return yolo_labels


# -------------------------------
# Keyboard callbacks
# -------------------------------
def on_press(key):
    global capture_flag, STOP
    try:
        if key.char == "c":
            capture_flag = True
        elif key.char == "d":
            STOP = True
            return False
    except AttributeError:
        pass


# -------------------------------
# Main loop
# -------------------------------
def main():
    global frame_id, capture_flag, STOP

    # Determine starting frame_id
    frame_id = get_frame_id(IMAGES_DIR)
    print(f"Starting at frame_id = {frame_id}")

    # Activate window once
    win = get_window()
    win.activate()
    time.sleep(0.2)  # allow focus
    bbox = get_window_info(win)

    # Prepare screenshot tool
    sct = mss.mss()

    print("Controls:")
    print("  C = capture + annotate")
    print("  D = quit")

    # Start keyboard listener (non-blocking)
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Main loop runs in main thread
    while not STOP:
        if capture_flag:
            capture_flag = False
            rgb_frame, bgr_frame = capture_frame(sct, bbox)
            labels = annotate_frame(bgr_frame)
            if labels:
                save_frame(rgb_frame, frame_id)
                save_labels(labels, frame_id)
                print(f"Saved frame {frame_id}")
                frame_id += 1
            else:
                print("Frame discarded")
        time.sleep(0.05)  # small sleep to avoid busy loop

    listener.join()
    print("Exiting.")


if __name__ == "__main__":
    main()
