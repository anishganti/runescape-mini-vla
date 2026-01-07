from mlx_vlm import load, generate

model, processor = load("mlx-community/SmolVLM-256M-Instruct-bf16")
print(type(model))

from PIL import Image
image = Image.open("/Users/anishganti/runescape-mini-vla/data/mining/episode_000001/frames/frame_000003.png")

print(model.config)
outputs = generate(model, processor, image=image, prompt="Describe a <image>")
#outputs = model(input_ids, pixel_values, cache=prompt_cache, mask=mask, **kwargs)
#print(outputs)


#model is type mlx_vlm.models.idefics3.idefics3.Model
#mlx_vlm.models.idefics3.idefics3.Model is built on nn.Module 