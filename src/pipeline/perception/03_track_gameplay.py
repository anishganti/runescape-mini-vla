import os
import cv2 
import numpy as np
from ultralytics import YOLO

# ------------------- PARAMETERS -------------------
#BASE_PATH = "/Users/anishganti/runescape_mini_vla/data/objects/mines/lumbridge"
#VIDEO_PATH = "mining.mp4"
VIDEO_PATH = "/Users/anishganti/runescape_mini_vla/data/mines/lumbridge/video1.mp4"
OUTPUT_DIR = "yolo_labels"
CLASS_ID = 0  # If you only have orne class (ore)
TRACKER_TYPE = "CSRT"  # Good for accuracy; MOSSE is faster but less accurate
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return False, None
    return True, cap

def load_frames(cap, window_size=20): 
    frames = []

    while True: 
        ok, frame = cap.read()
        if not ok: 
            return ok, frames

        frames.append(frame)
        if len(frames) == window_size:
            return ok, frames
        
def detect_frames(model, frames, batch_size=4):
    y_boxes = []

    for i in range(0, len(frames), batch_size):
        
        batch = frames[i:i+batch_size]
        results = model(batch, conf=0.4, iou=0.5, verbose=False)
        
        for result in results: 
            res = []
            boxes = result.boxes.xywh      
            boxes = boxes.cpu().numpy()

            for x, y, dx, dy in boxes:
                res.append((int(x), int(y), int(dx), int(dy)))
            y_boxes.append(res)

    return y_boxes

def track_frames(frames, trackers, idx=0): 
    t_boxes = [None for i in range(len(frames))]

    if trackers is None:
        bboxes, trackers = start_tracking(frames[idx])
        t_boxes[idx] = bboxes
        idx+=1

    t_boxes[idx:] = [trackers.update(frame)[1] for frame in frames[idx:]]

    return trackers, t_boxes

def get_coord(box, form):
    x,y,w,h = box[0], box[1], box[2], box[3]
    if form == "YOLO":
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
    else: 
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)
    return x1,y1,x2,y2

def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a[0], box_a[1], box_a[2], box_a[3]
    bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[2], box_b[3]

    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    inter_width = max(0, x_right - x_left)
    inter_height = max(0, y_bottom - y_top)
    inter_area = inter_width * inter_height

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    iou_value = inter_area / union_area if union_area > 0 else 0
    return iou_value

def greedy_match(tracker_boxes, yolo_boxes, iou_threshold=0.7):
    # Convert boxes to xyxy
    tracker_xyxy = [get_coord(b, "Tracker") for b in tracker_boxes]
    yolo_xyxy = [get_coord(b, "YOLO") for b in yolo_boxes]

    # IoU matrix
    iou_matrix = np.zeros((len(tracker_xyxy), len(yolo_xyxy)))
    for i, t_box in enumerate(tracker_xyxy):
        for j, y_box in enumerate(yolo_xyxy):
            iou_matrix[i, j] = iou(t_box, y_box)

    matched_tracker = []
    matched_yolo = []
    unmatched_tracker = set(range(len(tracker_xyxy)))
    unmatched_yolo = set(range(len(yolo_xyxy)))

    # Greedy matching
    while iou_matrix.size > 0:
        max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        i, j = max_idx
        if iou_matrix[i, j] < iou_threshold:
            break  # no matches above threshold

        # Add match
        matched_tracker.append(tracker_boxes[i])
        matched_yolo.append(yolo_boxes[j])
        unmatched_tracker.discard(i)
        unmatched_yolo.discard(j)

        # Remove matched row and column
        iou_matrix = np.delete(iou_matrix, i, axis=0)
        iou_matrix = np.delete(iou_matrix, j, axis=1)

        # Adjust indices for remaining unmatched sets
        unmatched_tracker = {idx if idx < i else idx-1 for idx in unmatched_tracker}
        unmatched_yolo = {idx if idx < j else idx-1 for idx in unmatched_yolo}

    # Collect unmatched boxes
    unmatched_tracker_boxes = [tracker_boxes[i] for i in unmatched_tracker]
    unmatched_yolo_boxes = [yolo_boxes[j] for j in unmatched_yolo]

    return len(unmatched_tracker) + len(unmatched_yolo_boxes) > 0

def compare_bounding_boxes(t_boxes, y_boxes):
    statuses = []

    for t_box, y_box in zip(t_boxes, y_boxes): 
        if len(t_box) != len(y_box) or not greedy_match(t_box, y_box): 
            statuses.append(False)
        else: 
            statuses.append(True)
    
    return statuses

def validate_frames(statuses): 
    failed_frame_id = None
    failed_frame_cnt = 0
    failed_in_a_row = 3

    for i in range(len(statuses)): 
        if not statuses[i]: 
            failed_frame_cnt+=1
            failed_in_a_row+=1

            if failed_frame_id is None: 
                failed_frame_id = i
        else: 
            failed_in_a_row = 0
        
        if failed_frame_cnt == 5 or failed_in_a_row == 3: 
            break

    return failed_frame_id

def repair_frames(frames, trackers, t_boxes, y_boxes):
    while True: 
        statuses = compare_bounding_boxes(t_boxes, y_boxes)
        print(statuses)
        fail_idx = validate_frames(statuses)

        if fail_idx is None: 
            return trackers, t_boxes

        trackers = track_frames(frames, None, fail_idx)   

def calc_frame_diff(frame1, frame2, threshold=.01): 
    diff = np.mean(np.abs(frame1.astype(float) - frame2.astype(float)) / 255)
    return diff > threshold

def prune_frames(frames, t_boxes): 
    idx = 0
    while True: 
        if idx == len(frames):
            return frames, t_boxes

        for i in range(idx+1, len(frames)): 
            if calc_frame_diff(frames[idx], frames[i]): 
                del frames[i], t_boxes[i]

        idx+=1

def save_frames(frames, bboxes):
    for idx, (frame, bbox) in enumerate(zip(frames, bboxes)):
        cv2.imwrite(f"/Users/anishganti/runescape_mini_vla/data/mines/lumbridge/images/{idx}.png", frame)

        with open(f"/Users/anishganti/runescape_mini_vla/data/mines/lumbridge/labels/{idx}.txt", "w") as f:
            for box in bbox:
                f.write(" ".join(map(str, box)) + "\n")

def annotate_frame(frame):
    bboxes = []
    while True:
        bbox = cv2.selectROI("Select Ore (press ENTER/SPACE, ESC to finish)", frame, False)
        if bbox == (0,0,0,0):
            break
        bboxes.append(bbox)

    cv2.destroyAllWindows()
    return bboxes

def create_tracker(frame, bboxes):
    trackers = cv2.legacy.MultiTracker_create()
    for bbox in bboxes:
        tracker = cv2.legacy.TrackerCSRT_create()
        trackers.add(tracker, frame, bbox)
    return trackers

def start_tracking(frame): 
    bboxes = annotate_frame(frame)
    trackers = create_tracker(frame, bboxes)
    return bboxes, trackers

def draw_xywh(frame, boxes, form, color, thickness=2):
    for x, y, w, h in boxes:
        x1, y1, x2, y2 = get_coord(box, form)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

def replay_video(frames, boxes_a, boxes_b, fps=30):
    delay = int(1000 / fps)

    for i, frame in enumerate(frames):
        vis = frame.copy()

        if i < len(boxes_a):
            draw_xywh(vis, boxes_a[i], 'YOLO', color=(0, 255, 0))  # green
        if i < len(boxes_b):
            draw_xywh(vis, boxes_b[i], 'Tracker', color=(0, 0, 255))  # red

        cv2.imshow("Replay", vis)
        if cv2.waitKey(delay) & 0xFF == 27:  # ESC
            break

    cv2.destroyAllWindows()

def track_pipeline():
    # cv2 Video Capture loads video frame-by-frame, so we loop until we run through all the frames
    trackers = None
    ok, cap = load_video(VIDEO_PATH)
    model = YOLO("/Users/anishganti/runescape_mini_vla/runs/detect/runescape_ore/weights/best.pt")

    while ok:
        ok, frames = load_frames(cap)
        y_boxes = detect_frames(model, frames)
        trackers, t_boxes = track_frames(frames, trackers)
        #replay_video(frames, y_boxes, t_boxes)
        trackers, t_boxes = repair_frames(frames, trackers, t_boxes, y_boxes)
        frames, boxes = prune_frames(frames, t_boxes)
        save_frames(frames, boxes)

track_pipeline()

