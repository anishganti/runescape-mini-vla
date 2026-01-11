import os
import json
import torch
from PIL import Image

base_dir = "/Users/anishganti/runescape_mini_vla/data/mining"

def get_episode_id(path):
    episodes = [name for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))]
    if not episodes:
        return 1
    return max(int(ep[-6:]) for ep in episodes) + 1

def get_episodes(path):
    episodes = [
        episode for episode in os.listdir(path)
        if os.path.isdir(os.path.join(path, episode))
    ]
    
    return episodes

def load_embeddings(episode):
    file_name = f"/Users/anishganti/runescape_mini_vla/data/mining/{episode}/{episode}_embeddings.pt"
    embeddings = torch.load(file_name)
    embeddings = torch.stack(embeddings)
    return embeddings

def load_actions(episode):
    file_name = f"/Users/anishganti/runescape_mini_vla/data/mining/{episode}/{episode}.json"

    with open(file_name, "r", encoding="utf-8") as file_handle:
        events = json.load(file_handle)
        actions = events['events']        
        return actions

def load_images(episode): 
    images_dir = f"{base_dir}/{episode}/frames"
    images = [
        load_image(os.path.join(images_dir, image))
        for image in sorted(os.listdir(images_dir))
    ]
    return images
