from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-seg-pf.pt")

# Run prediction. No prompts required.
results = model.predict("messy-desk.jpg")

# Show results
results[0].show()