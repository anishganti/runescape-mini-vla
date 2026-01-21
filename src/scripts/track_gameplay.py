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

def calc_box_diff(t_box, y_box, threshold=5): 
    diff = sum(abs(t_box[i] - y_box[i]) for i in range(4))
    return diff > threshold

def compare_bounding_boxes(t_boxes, y_boxes):
    statuses = []

    for t_box, y_box in zip(t_boxes, y_boxes): 
        if len(t_box) != len(y_box) or calc_box_diff(t_box, y_box): 
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

def repair_frames(frames, t_boxes, y_boxes):
    while True: 
        statuses = compare_bounding_boxes(t_boxes, y_boxes)
        fail_idx = validate_frames(statuses)

        if fail_idx is None: 
            return trackers, t_boxes

        trackers = track_frames(frames, None, fail_idx)   

def calc_frame_diff(frame1, frame2): 
    mean_diff = np.mean(np.abs(frame1 - frame2))
    return mean_diff < threshold

def prune_frames(frames, t_boxes): 
    idx = 0
    while True: 
        if idx == len(frames):
            return frames, t_boxes

        for i in range(idx+1, len(frames)): 
            if calc_frame_diff(frames[idx], frames[i]): 
                del frames[i], t_boxes[i]

        idx+=1

def save_frames(frames, boxes):
    pass

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

def draw_xywh(frame, boxes, color, thickness=2):
    for x, y, w, h in boxes:
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

def replay_video(frames, boxes_a, boxes_b, fps=30):
    delay = int(1000 / fps)

    for i, frame in enumerate(frames):
        vis = frame.copy()

        if i < len(boxes_a):
            draw_xywh(vis, boxes_a[i], color=(0, 255, 0))  # green
        if i < len(boxes_b):
            draw_xywh(vis, boxes_b[i], color=(0, 0, 255))  # red

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
        print(len(t_boxes))
        print(len(y_boxes))
        replay_video(frames, y_boxes, t_boxes)
        #trackers, t_boxes = repair_frames(frames, t_boxes, y_boxes)
        #frames, boxes = prune_frames(frames, t_boxes)
        #save_frames(frames, boxes)

track_pipeline()

