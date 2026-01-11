import mss
import torch
import os
import numpy as np
import pywinctl as pwc
from src.scripts.extract_embeddings import process_images
from src.scripts.train_action_policy_head import load_model, pool_embeddings
from src.scripts.capture_frames import capture_frame, get_window

def predict_action(embeddings):
    model = load_model()
    frame = capture_frame()
    embeddings = process_images(frame)
    embeddings = pool_embeddings(embeddings.cpu())
    a,k,x,y = model(embeddings)
    a = a.argmax(dim=1)
    k = k.argmax(dim=1)
    x = x.argmax(dim=1)
    y = y.argmax(dim=1)
    return a,k,x,y

def convert_bucket_to_coord(x,y): 
    MOUSE_BUCKETS = 20
    win = get_window()
    bbox = get_window_info(win)
    return (x * bbox["width"] / MOUSE_BUCKETS + bbox["left"], y * bbox["height"] / MOUSE_BUCKETS + bbox["top"])