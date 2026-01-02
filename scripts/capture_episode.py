import time
import json
from threading import Thread
from collections import deque
import os

import pywinctl as pwc
import mss
import numpy as np
from PIL import Image
from pynput import keyboard, mouse

# -------------------------------
# Config
# -------------------------------
BASE_DIR = "data/mining"
KEY_MAP = {"up": 0, "down": 1, "left": 2}
BUTTON_MAP = {"left": 0}
MOUSE_BUCKETS = 20  # 20x20 grid
IDLE_INTERVAL = 3.0  # seconds per idle frame

# -------------------------------
# Helpers
# -------------------------------
def get_episode_id(path):
    episodes = [
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]
    if len(episodes) == 0:
        return 1
    episode_ids = [int(ep[-6:]) for ep in episodes]
    return max(episode_ids) + 1

def get_time():
    return time.perf_counter()

def get_window():
    try:
        return pwc.getWindowsWithTitle("Old School RuneScape")[0]
    except IndexError:
        raise Exception("RuneScape window not found. Make sure it's open.")

def get_window_info(win):
    return {"left": win.left, "top": win.top, "width": win.width, "height": win.height}

def save_frame(frame, frame_id, frame_dir):
    path = os.path.join(frame_dir, f"frame_{frame_id:06d}.png")
    Image.fromarray(frame).save(path)
    return path

def convert_bgra_to_rgb(frame):
    frame = frame[:, :, :3]
    frame = frame[:, :, ::-1]
    return frame

# -------------------------------
# Action capture with quantization
# -------------------------------
def capture_action(get_time_func, actions_deque, move_buffer, frame_queue, window_info):
    def log_action(time, type_, event, key=None, x=None, y=None):
        actions_deque.append({
            "time": time,
            "type": type_,   # 0=key,1=mouse button,2=mouse move
            "event": event,  # 0=release,1=press,2=move
            "key": key,
            "x": x,
            "y": y
        })

    def quantize(x, y):
        bucket_x = int((x - window_info["left"]) / window_info["width"] * MOUSE_BUCKETS)
        bucket_y = int((y - window_info["top"]) / window_info["height"] * MOUSE_BUCKETS)
        bucket_x = max(0, min(MOUSE_BUCKETS - 1, bucket_x))
        bucket_y = max(0, min(MOUSE_BUCKETS - 1, bucket_y))
        return bucket_x, bucket_y

    # --- Keyboard ---
    def on_key_press(k):
        if hasattr(k, "char"):
            return
        if k.name in KEY_MAP:
            log_action(get_time_func(), 0, 1, key=KEY_MAP[k.name])
            frame_queue.append(True)

    def on_key_release(k):
        if hasattr(k, "char"):
            return
        if k.name in KEY_MAP:
            log_action(get_time_func(), 0, 0, key=KEY_MAP[k.name])
            frame_queue.append(True)

    # --- Mouse ---
    def on_click(x, y, button, pressed):
        if button.name in BUTTON_MAP:
            if move_buffer["time"] is not None:
                bx, by = quantize(move_buffer["x"], move_buffer["y"])
                log_action(move_buffer["time"], 2, 2, x=bx, y=by)
                move_buffer["time"] = None
            log_action(get_time_func(), 1, int(pressed), key=BUTTON_MAP[button.name])
            frame_queue.append(True)

    def on_move(x, y):
        move_buffer["time"] = get_time_func()
        move_buffer["x"] = x
        move_buffer["y"] = y

    k_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    m_listener = mouse.Listener(on_click=on_click, on_move=on_move)

    k_listener.start()
    m_listener.start()

    return k_listener, m_listener

# -------------------------------
# Frame capture (action + sparse idle)
# -------------------------------
def capture_frames(frames_deque, frame_dir, bbox, stop_flag, frame_queue):
    import mss
    frame_id = 0
    last_capture_time = 0
    with mss.mss() as sct:
        while not stop_flag["stop"]:
            t = time.perf_counter()
            # Capture if action occurred or idle interval reached
            if frame_queue or (t - last_capture_time >= IDLE_INTERVAL):
                frame_queue.clear()
                sct_img = sct.grab(bbox)
                frame = convert_bgra_to_rgb(np.array(sct_img))
                save_frame(frame, frame_id, frame_dir)
                frames_deque.append(t)
                frame_id += 1
                last_capture_time = t
            else:
                time.sleep(0.05)

# -------------------------------
# Main
# -------------------------------
def main():
    EPISODE_ID = get_episode_id(BASE_DIR)
    FRAME_DIR = os.path.join(BASE_DIR, f"episode_{EPISODE_ID:06d}/frames")
    LOG_FILE = os.path.join(BASE_DIR, f"episode_{EPISODE_ID:06d}/episode_{EPISODE_ID:06d}.json")
    os.makedirs(FRAME_DIR, exist_ok=True)

    frames = deque()
    actions = deque()
    move_buffer = {"time": None, "x": None, "y": None}
    frame_queue = deque()

    win = get_window()
    bbox = get_window_info(win)
    stop_flag = {"stop": False}

    # Start threads
    frame_thread = Thread(target=capture_frames, args=(frames, FRAME_DIR, bbox, stop_flag, frame_queue), daemon=True)
    frame_thread.start()

    k_listener, m_listener = capture_action(get_time, actions, move_buffer, frame_queue, bbox)

    print(f"Started episode {EPISODE_ID}. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping capture...")
        stop_flag["stop"] = True
        frame_thread.join()
        k_listener.stop()
        m_listener.stop()

        # Save log
        log = {
            "episode": EPISODE_ID,
            "bbox": bbox,
            "frames": list(frames),
            "actions": list(actions)
        }
        with open(LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)

        print(f"Saved {len(frames)} frames and {len(actions)} actions to {LOG_FILE}")

if __name__ == "__main__":
    main()
