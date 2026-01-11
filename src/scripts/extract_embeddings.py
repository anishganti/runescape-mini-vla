import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
import time
import os
from src.scripts.load_data import get_episodes, load_images, load_embeddings


DEVICE = "mps" if torch.mps.is_available() else "cpu"
base_dir = "/Users/anishganti/runescape_mini_vla/src/data/mining"

def init_vlm():
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        dtype=torch.float32
    ).to(DEVICE)

    return processor, model

def format_input(images, processor):
    prompts = []
    for _ in images:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "mine"}]
            },
        ]
        prompts.append(processor.apply_chat_template(messages, add_generation_prompt=True))

    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = inputs.to(DEVICE)
    return inputs

def run_model(inputs, model):
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    return outputs

def extract_embeddings(outputs):
    return outputs.hidden_states[-1]

def embedding_file_exists(episode):
    file_name = f"/Users/anishganti/runescape-mini-vla/notebooks/{episode}_embeddings.pt"
    return os.path.isfile(file_name)

def process_images(images):
    processor, model = init_vlm()
    inputs = format_input(images, processor)
    outputs = run_model(inputs, model)
    embeddings = extract_embeddings(outputs)
    return embeddings

def main():
    episodes = get_episodes(base_dir)
    processor, model = init_vlm()

    for ep in episodes: 
        if embedding_file_exists(ep):
            continue

        batch_size = 4
        all_embeddings = []
        images = load_images(ep)
        num_images = len(images)

        for i in range(0, num_images, batch_size):
            image_batch = images[i:min(num_images, i+batch_size)]
            embeddings = forward(image_batch, model, processor)
            all_embeddings.extend(embeddings.cpu())

        torch.save(all_embeddings, f"{base_dir}/{ep}/{ep}_embeddings.pt")
        
