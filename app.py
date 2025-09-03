import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


model = load_model("face_mask_detection.h5") 

st.title("😷 Face Mask Detection App")
st.write("Upload an image and the model will predict if the person is wearing a mask or not.")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    
    input_image = np.array(image)

    
    input_image_resized = cv2.resize(input_image, (128, 128))
    input_image_scaled = input_image_resized / 255.0
    input_image_reshaped = np.reshape(input_image_scaled, (1, 128, 128, 3))

    
    prediction = model.predict(input_image_reshaped)
    pred_label = np.argmax(prediction)

    
    if pred_label == 1:
        st.success("✅ The person in the image is **wearing a mask**.")
    else:
        st.error("❌ The person in the image is **not wearing a mask**.")

    st.write("### Prediction probabilities:")
    st.write(prediction)
