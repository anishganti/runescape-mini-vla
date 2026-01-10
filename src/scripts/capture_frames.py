import time
import os
from collections import deque

import pywinctl as pwc
import mss
import numpy as np
from PIL import Image
from pynput import keyboard, mouse
import json

# -------------------------------
# Config / Top-level shared variables
# -------------------------------
BASE_DIR = "data/mining"
IDLE_INTERVAL = 3.0  # seconds

# Numeric action/key maps
ACTIONS = {"press": 0, "release": 1, "wait": 2}
KEYS = {"mouse": 0, "left": 1, "up": 2, "down": 3}
MOUSE_BUCKETS = 20

# Shared state
events = deque()
frame_id = 0
last_capture_time = 0
first_action_done = False
stop_flag = {"stop": False}
FRAME_DIR = None
bbox = None
sct = None

# -------------------------------
# Helpers
# -------------------------------
def get_episode_id(path):
    episodes = [name for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))]
    if not episodes:
        return 1
    return max(int(ep[-6:]) for ep in episodes) + 1

def get_window():
    wins = pwc.getWindowsWithTitle("Old School RuneScape")
    if not wins:
        raise RuntimeError("RuneScape window not found")
    return wins[0]

def get_window_info(win):
    return {"left": win.left, "top": win.top, "width": win.width, "height": win.height}

def convert_bgra_to_rgb(frame):
    return frame[:, :, :3][:, :, ::-1]

def save_frame(frame, frame_id, frame_dir):
    path = os.path.join(frame_dir, f"frame_{frame_id:06d}.png")
    Image.fromarray(frame).save(path)

def bucketize(x, y, bbox):
    bx = int((x - bbox["left"]) / bbox["width"] * MOUSE_BUCKETS)
    by = int((y - bbox["top"]) / bbox["height"] * MOUSE_BUCKETS)
    bx = max(0, min(MOUSE_BUCKETS - 1, bx))
    by = max(0, min(MOUSE_BUCKETS - 1, by))
    return bx, by

def generate_action(action_type, key=None, x=None, y=None, bbox=None):
    out = {"a": ACTIONS[action_type]}
    if key is not None:
        out["k"] = KEYS[key]
    if x is not None and y is not None and bbox is not None:
        bx, by = bucketize(x, y, bbox)
        out["x"] = bx
        out["y"] = by
    return out

def capture_frame_and_action(action):
    """Capture the current frame + append the action to events deque."""
    global frame_id, last_capture_time, sct
    img = sct.grab(bbox)
    frame = convert_bgra_to_rgb(np.array(img))
    save_frame(frame, frame_id, FRAME_DIR)
    events.append(action)
    frame_id += 1
    last_capture_time = time.perf_counter()

# -------------------------------
# Input handlers
# -------------------------------
def on_key_press(k):
    global first_action_done
    if k == keyboard.Key.esc:
        stop_flag["stop"] = True
        return False
    if hasattr(k, "name") and k.name in ["up", "down", "left"]:
        act = generate_action("press", key=k.name)
        capture_frame_and_action(act)
        first_action_done = True

def on_key_release(k):
    global first_action_done
    if hasattr(k, "name") and k.name in ["up", "down", "left"]:
        act = generate_action("release", key=k.name)
        capture_frame_and_action(act)
        first_action_done = True

def on_click(x, y, button, pressed):
    global first_action_done
    if button.name == "left":
        act = generate_action(
            "press" if pressed else "release",
            key="mouse",
            x=x, y=y,
            bbox=bbox
        )
        capture_frame_and_action(act)
        first_action_done = True

def capture_frames():
    win = get_window()
    win.activate()
    sct = mss.mss()

    bbox = get_window_info(win)
    img = sct.grab(bbox)
    frame = convert_bgra_to_rgb(np.array(img))
    return frame

def capture_episode():
    global FRAME_DIR, bbox, sct, last_capture_time

    EPISODE_ID = get_episode_id(BASE_DIR)
    EP_DIR = os.path.join(BASE_DIR, f"episode_{EPISODE_ID:06d}")
    FRAME_DIR = os.path.join(EP_DIR, "frames")
    LOG_FILE = os.path.join(EP_DIR, f"episode_{EPISODE_ID:06d}.json")
    os.makedirs(FRAME_DIR, exist_ok=True)

    # Bring window to front
    win = get_window()
    win.activate()
    time.sleep(0.5)
    bbox = get_window_info(win)

    last_capture_time = time.perf_counter()
    sct = mss.mss()

    # Start input listeners
    keyboard.Listener(on_press=on_key_press, on_release=on_key_release).start()
    mouse.Listener(on_click=on_click).start()

    print(f"Started episode {EPISODE_ID}. Press ESC to stop.")

    # Idle capture loop
    try:
        while not stop_flag["stop"]:
            t = time.perf_counter()
            if first_action_done and t - last_capture_time >= IDLE_INTERVAL:
                act = generate_action("wait")
                capture_frame_and_action(act)
            time.sleep(0.01)
    finally:
        with open(LOG_FILE, "w") as f:
            json.dump({"events": list(events)}, f, indent=2)
        print(f"Saved {len(events)} frames to {LOG_FILE}")

if __name__ == "__main__":
    main()
