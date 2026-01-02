# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/Users/anishganti/runescape-mini-vla/data/mining/episode_000001/frames/frame_000006.png"},
            {"type": "text", "text": "Describe the image?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))