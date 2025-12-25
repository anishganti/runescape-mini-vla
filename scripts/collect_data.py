import mss
import time
import os
import json
from threading import Thread
from PIL import Image
import numpy as np
import pywinctl as pwc

FPS = 1                  
FRAME_DIR = "data/woodcut_lumbridge"     
LOG_FILE = "logs/events.json"  

events = []


win = pwc.getWindowsWithTitle("Old School RuneScape")[0]
bbox = (win.left, win.top, win.right, win.bottom)
print(bbox)


def get_current_timestamp():
    return time.perf_counter()  

def save_frame(frame, frame_id): 
    path = os.path.join(FRAME_DIR, f"frame_{frame_id:05d}.png")
    img = Image.fromarray(frame)
    img.save(path)
    return path

def capture_frames():
    with mss.mss() as sct:
        frame_id = 0
        while True:
            timestamp = get_current_timestamp()
            #win.activate()
            screenshot = sct.grab({
                "left": bbox[0],
                "top": bbox[1],
                "width": bbox[2]-bbox[0],
                "height": bbox[3]-bbox[1]
            })
            frame = np.array(screenshot)
            path = save_frame(frame, frame_id)

            events.append({
                "type": "frame",
                "timestamp": timestamp,
                "frame_id": frame_id,
                "frame_path": path
            })

            frame_id += 1

            dt = 1/FPS - (get_current_timestamp() - timestamp)
            if dt > 0:
                time.sleep(dt)

def capture_actions():
    return


def process_image(): 
    return


capture_thread = Thread(target=capture_frames, daemon=True)
capture_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Saving event log...")
    with open(LOG_FILE, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Saved {len(events)} events")