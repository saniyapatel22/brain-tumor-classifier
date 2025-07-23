
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

model = load_model('models/custom_cnn_model.h5')
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

st.title("Brain Tumor MRI Classifier")

uploaded_file = st.file_uploader("Upload MRI Image", type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    pred_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    st.markdown(f"### Prediction: {pred_class} ({confidence:.2%} confidence)")
