import mss
import torch
import os
import numpy as np
import pywinctl as pwc
from src.scripts.extract_embeddings import init_vlm, forward
from src.scripts.train_action_policy_head import ActionPolicyHead, pool_embeddings

def get_window():
    wins = pwc.getWindowsWithTitle("Old School RuneScape")
    if not wins:
        raise RuntimeError("RuneScape window not found")
    return wins[0]

def get_window_info(win):
    return {"left": win.left, "top": win.top, "width": win.width, "height": win.height}

def convert_bgra_to_rgb(frame):
    return frame[:, :, :3][:, :, ::-1]

def capture_frame():
    win = get_window()
    win.activate()
    sct = mss.mss()

    bbox = get_window_info(win)
    img = sct.grab(bbox)
    frame = convert_bgra_to_rgb(np.array(img))
    return frame

def convert_bucket_to_coord(x,y): 
    MOUSE_BUCKETS = 20
    win = get

def extract_embeddings(frame):
    processor, model = init_vlm()
    embeddings = forward([frame], model, processor)
    return embeddings

def run_action(embeddings):
    model = ActionPolicyHead()
    model.load_state_dict(torch.load("/Users/anishganti/runescape_mini_vla/src/models/mlp/checkpoint.pt"))
    model.eval()   # important
    embeddings = pool_embeddings(embeddings.cpu())
    a,k,x,y = model.forward(embeddings)
    return a,k,x,y


frame = capture_frame()
embeddings = extract_embeddings(frame)
print(embeddings)
print(embeddings.shape)
a,k,x,y = run_action(embeddings)
print(a.argmax(dim=1))
print(k.argmax(dim=1))
print(x.argmax(dim=1))
print(y.argmax(dim=1))