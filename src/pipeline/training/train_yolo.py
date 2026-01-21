# yolov8_train.py
from ultralytics import YOLO
import os

# -------------------------------
# Config
# -------------------------------
DATA_DIR = "/Users/anishganti/runescape_mini_vla/data/mines/lumbridge"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

# YOLO requires a dataset YAML file
DATA_YAML = os.path.join(DATA_DIR, "dataset.yaml")
PRETRAINED_WEIGHTS = "yolov8n.pt"  # nano model, small and fast
EPOCHS = 50
BATCH_SIZE = 4  # adjust to GPU memory
IMG_SIZE = 640

# -------------------------------
# Step 1: Create dataset.yaml
# -------------------------------
# YOLOv8 expects a YAML like this:
# train: path/to/images
# val: path/to/images
# nc: 1
# names: ['ore']

if not os.path.exists(DATA_YAML):
    yaml_content = f"""
train: {IMAGES_DIR}
val: {IMAGES_DIR}
nc: 1
names: ['ore']
"""
    with open(DATA_YAML, "w") as f:
        f.write(yaml_content)
    print(f"Created dataset YAML at {DATA_YAML}")

# -------------------------------
# Step 2: Load pretrained model
# -------------------------------
model = YOLO(PRETRAINED_WEIGHTS)

# -------------------------------
# Step 3: Train
# -------------------------------
# Very small dataset → use low epochs
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    name="runescape_ore",
    pretrained=True
)

# -------------------------------
# Step 4: Save best model
# -------------------------------
# Ultralytics automatically saves to ./runs/detect/run_name
print("Training complete. Best model saved in:")
print(os.path.join("runs", "detect", "runescape_ore"))
