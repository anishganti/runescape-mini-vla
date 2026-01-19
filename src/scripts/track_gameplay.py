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

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap

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

    for i in range(0, len(frames), batch_size)
        batch = frames[i:i+batch_size]
        y_boxes.append(model(batch, conf=0.4, iou=0.5, device=0, verbose=False))

    return y_boxes

def track_frames(frames, tracker, idx=0): 
    tboxes = [None * len(frames)]

    if tracker is None:
        bboxes, tracker = start_tracking(frames[idx])
        tboxes[idx] = bboxes
        idx++

    t_boxes[idx:] = [trackers.update(frame) for frame in frames[idx:]]

    return tracker, t_boxes

def compare_bounding_boxes(t_boxes, y_boxes):
    pass

def repair_frames(frames, t_boxes, y_boxes, trackers):
    while True: 
        statuses = compare_bounding_boxes(t_boxes, y_boxes)
        fail_idx = validate_frames(statuses)

        if fail_idx is None: 
            return trackers, t_boxes

        trackers = track_frames(frames, None, fail_idx)        

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

def track_pipeline():
    # cv2 Video Capture loads video frame-by-frame, so we loop until we run through all the frames
    trackers = None
    ok, cap = load_video(VIDEO_PATH)
    model = YOLO("weights/game_detection.pt")

    while ok:
        ok, frames = load_frames(cap)
        y_boxes = detect_frames(model, frames)
        trackers, t_boxes = track_frames(frames, trackers)
        trackers, t_boxes = repair_frames(frames, t_boxes, y_boxes)
        frames, boxes = prune_frames(frames, t_boxes, y_boxes)
        save_frames(frames, boxes)




   


