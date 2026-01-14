import pywinctl as pwc
import mss
import cv2
import time
import numpy as np

FPS = 30
DURATION_SEC = 5  # set None for infinite
VIDEO_PATH = "/Users/anishganti/runescape_mini_vla/data/objects/mines/lumbridge/mining.mp4"

def get_window():
    wins = pwc.getWindowsWithTitle("Old School RuneScape")
    if not wins:
        raise RuntimeError("RuneScape window not found")
    return wins[0]

def get_window_info(win):
    return {"left": win.left, "top": win.top, "width": win.width, "height": win.height}


def record():
    win = get_window()
    bbox = get_window_info(win)
    win.activate()

    with mss.mss() as sct:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            VIDEO_PATH, fourcc, FPS, (bbox['width'], bbox['height'])
        )

        frame_interval = 1.0 / FPS
        start_time = time.time()
        frame_count = 0

        print("Recording started... Press Ctrl+C to stop.")

        try:
            while True:
                loop_start = time.time()

                img = sct.grab(bbox)
                frame = np.array(img)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                out.write(frame)

                frame_count += 1

                if DURATION_SEC is not None:
                    if time.time() - start_time >= DURATION_SEC:
                        break

                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_interval - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("Recording stopped by user.")

        finally:
            out.release()
            print(f"Saved {frame_count} frames to {VIDEO_PATH}")

record()