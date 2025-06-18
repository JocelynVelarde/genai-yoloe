from ultralytics import YOLOE
import streamlit as st 

def detect_classes(class_names, image_path): 
    # Initialize a YOLOE model
    model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes
    # test_classes = ["cat", "dog"]
    
    # Set text prompt to detect person and bus. You only need to do this once after you load the model.
    model.set_classes(class_names, model.get_text_pe(class_names))

    # Run detection on the given image
    results = model.predict(image_path)

    # Show results
    #st.image()
    return results[0].plot()