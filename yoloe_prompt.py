from ultralytics import YOLOE
from PIL import Image
import numpy as np

def detect_classes(class_names, image_file):
    # Ensure class_names is a list
    if isinstance(class_names, str):
        class_list = [c.strip() for c in class_names.split(",") if c.strip()]
    else:
        class_list = class_names

    # Convert uploaded file to PIL Image
    image = Image.open(image_file).convert("RGB")

    model = YOLOE("yoloe-11l-seg.pt")
    model.set_classes(class_list, model.get_text_pe(class_list))
    results = model.predict(image)
    return results[0].plot()