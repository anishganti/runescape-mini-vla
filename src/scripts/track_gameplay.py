import os
import cv2 
import numpy as np

# ------------------- PARAMETERS -------------------
#BASE_PATH = "/Users/anishganti/runescape_mini_vla/data/objects/mines/lumbridge"
#VIDEO_PATH = "mining.mp4"
VIDEO_PATH = "/Users/anishganti/runescape_mini_vla/data/objects/mines/lumbridge/mining.mp4"
OUTPUT_DIR = "yolo_labels"
CLASS_ID = 0  # If you only have orne class (ore)
TRACKER_TYPE = "CSRT"  # Good for accuracy; MOSSE is faster but less accurate
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize video
cap = cv2.VideoCapture(VIDEO_PATH)
ok, frame = cap.read()
if not ok:
    raise RuntimeError("Cannot read video.")

# Select ROIs manually
bboxes = []
while True:
    bbox = cv2.selectROI("Select Ore (press ENTER/SPACE, ESC to finish)", frame, False)
    if bbox == (0,0,0,0):
        break
    bboxes.append(bbox)

cv2.destroyAllWindows()

# Initialize MultiTracker
trackers = cv2.legacy.MultiTracker_create()
for bbox in bboxes:
    if TRACKER_TYPE == "CSRT":
        tracker = cv2.legacy.TrackerCSRT_create()
    elif TRACKER_TYPE == "KCF":
        tracker = cv2.legacy.TrackerKCF_create()
    elif TRACKER_TYPE == "MOSSE":
        tracker = cv2.legacy.TrackerMOSSE_create()
    else:
        tracker = cv2.legacy.TrackerCSRT_create()
    trackers.add(tracker, frame, bbox)

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    ok, boxes = trackers.update(frame)

    # Save YOLO-format labels
    h, w, _ = frame.shape
    label_path = os.path.join(OUTPUT_DIR, f"{frame_idx:06d}.txt")
    with open(label_path, "w") as f:
        for box in boxes:
            x, y, bw, bh = box
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            width = bw / w
            height = bh / h
            f.write(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # Optional: visualize tracking
    for box in boxes:
        x, y, bw, bh = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
