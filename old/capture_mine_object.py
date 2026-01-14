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
FRAME_DIR = "/Users/anishganti/runescape_mini_vla/data/objects/mines/lumbridge"
MOUSE_BUCKETS = 20
STOP = False
frame_id = 0    

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

def capture_frame():
    win = get_window()
    win.activate()
    sct = mss.mss()

    bbox = get_window_info(win)
    img = sct.grab(bbox)
    frame = convert_bgra_to_rgb(np.array(img))
    return frame

def on_press(key):
    global frame_id
    try: 
        if key.char == 'c': 
            frame = capture_frame()
            labels = annotate_frame(frame)
            save_frame(frame, frame_id, FRAME_DIR)
            save_labels(labels, frame_id, LABELS_DIR)
            frame_id += 1
        elif key.char == 'd': 
            stop = True
    except AttributeError:
        print(f'Special key pressed: {keyboard.Key}')

def on_release():
    pass

def main(): 
    print(f"Started capturing frames. Press D to stop.")

    with keyboard.Listener(
        on_press=on_press, 
        on_release=on_release
    ) as listener: 

        listener.join()

if __name__ == "__main__":
    main()