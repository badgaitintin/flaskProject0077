import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import io
import base64
import json
from datetime import datetime
import os

# Load the model
model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = torch.nn.Linear(1280, 3)  # Changed to 3 classes
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    return preprocess(img).unsqueeze(0)

# Convert image bytes to base64 string
def image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# File to store history
HISTORY_FILE = 'prediction_history.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        for i, item in enumerate(history):
            item['id'] = i
        return history
    return []

def save_history(history):
    for i, item in enumerate(history):
        item['id'] = i
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

# Load history at the start
history = load_history()

# Streamlit Interface
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌱 Plant Disease Classification")

uploaded_file = st.file_uploader("Upload an image...", type="jpg")

if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    img_tensor = preprocess_image(img_bytes)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)

    labels = ["Fungi/Bacteria", "Healthy", "Nutrient"]
    predicted_label = labels[predicted.item()]

    # Display image with a box shadow
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/jpg;base64,{image_to_base64(img_bytes)}" style="width: 80%; border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);" />
        </div>
        <h3 style="text-align: center; color: #333;">Predicted: {predicted_label}</h3>
        """,
        unsafe_allow_html=True
    )

    # Add new data to history
    new_item = {
        'id': len(history),
        'filename': uploaded_file.name,
        'label': predicted_label,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'img_data': image_to_base64(img_bytes)
    }
    history.append(new_item)
    save_history(history)

# Clear history button with confirmation
if st.button('Clear History'):
    history = []
    save_history(history)
    st.success("History cleared.")

# Display Prediction History
if history:
    st.write("### Prediction History")
    num_columns = 4
    cols = st.columns(num_columns)

    for i, item in enumerate(history):
        img_data = item.get('img_data')
        if img_data:
            with cols[i % num_columns]:
                card_html = f"""
                <div style="border:1px solid #ddd;padding:10px;margin-bottom:10px;border-radius:5px;box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);">
                    <img src="data:image/jpg;base64,{img_data}" style="width:100%;height:150px;object-fit:cover;border-radius:5px;">
                    <p style="margin-top:10px;"><strong>{item['filename']}</strong></p>
                    <p>Prediction: {item['label']}</p>
                    <p><small class="text-muted">Analyzed on: {item['timestamp']}</small></p>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
else:
    st.write("No prediction history available.")

# Instructions
st.markdown("### Instructions")
st.markdown(
    """
    1. Upload an image of a plant leaf.
    2. Wait for the model to classify the disease.
    3. View the prediction history below.
    """
)

# Run "streamlit run app.py" on terminal
