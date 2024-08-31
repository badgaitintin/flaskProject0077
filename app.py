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
        # Ensure each item has an 'id' field
        for i, item in enumerate(history):
            item['id'] = i
        return history
    return []

def save_history(history):
    # Ensure each item has an 'id' field before saving
    for i, item in enumerate(history):
        item['id'] = i
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

# Load history at the start
history = load_history()

# Streamlit Interface
st.title("Plant Disease Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    img_tensor = preprocess_image(img_bytes)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)

    labels = ["Fungi/Bacteria", "Healthy", "Nutrient"]
    predicted_label = labels[predicted.item()]

    # Display image
    st.image(uploaded_file, caption=f"Predicted: {predicted_label}", use_column_width=True)

    # Add new data to history
    new_item = {
        'id': len(history),
        'filename': uploaded_file.name,
        'label': predicted_label,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'img_data': image_to_base64(img_bytes)  # Ensure this is added
    }
    history.append(new_item)
    save_history(history)

# Display Prediction History
st.write("Prediction History")
num_columns = 4  # Adjust the number of columns here
cols = st.columns(num_columns)


# Loop through history to display cards in columns
for i, item in enumerate(history):
    img_data = item.get('img_data')  # Use .get() to avoid KeyError
    if img_data:  # Check if img_data exists
        with cols[i % num_columns]:  # Use modulo to distribute items across columns
            card_html = f"""
            <div style="border:1px solid #ddd;padding:10px;margin-bottom:10px;border-radius:5px;">
                <img src="data:image/jpg;base64,{img_data}" style="width:100%;height:150px;object-fit:cover;border-radius:5px;">
                <p><strong>{item['filename']}</strong></p>
                <p>Prediction: {item['label']}</p>
                <p><small class="text-muted">Analyzed on: {item['timestamp']}</small></p>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)


# Loop through history to display cards in columns
if st.button('Clear History'):
    history = []  # Clear the in-memory history
    save_history(history)  # Save the cleared history
    st.experimental_rerun()  # Rerun the app to reflect changes

# webapp is not ready
# run "streamlit run app.py" on terminal