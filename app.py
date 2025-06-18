import streamlit as st 
from yoloe_prompt import detect_classes

st.title("Test YOLOe out")

my_classes = st.text_input("Write the classes that you want to detect")

st.write('Upload an image to detect classes')
my_file = st.file_uploader('Upload an image to detect classes', 'jpg')

if st.button("Launch YOLOe"): 
    st.image(detect_classes(my_classes, my_file))
    st.success("Classes detected!")
else:
    st.warning("Please type an input")