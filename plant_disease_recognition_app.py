import streamlit as st
import requests

# Flask API URL
API_URL = "http://127.0.0.1:5000/predict"

st.title("Plant Disease Recognition")

uploaded_file = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        # Send image to Flask API
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            data = response.json()
            st.success(f"Prediction: {data['prediction']}")
            st.write(f"Confidence: {data['confidence'] * 100:.2f}%")
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")
