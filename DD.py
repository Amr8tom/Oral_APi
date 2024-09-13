from PIL import Image
import numpy as np
from keras import models
import streamlit as st

# Preprocessing function for the input image
def preprossing(image):
    image = image.resize((50, 50))
    image = np.asarray(image).astype("float32") / 255.0
    image = image.reshape((1,) + image.shape)
    return image

# Streamlit title for the web app
st.title("Oral Cancer Detection")

# Upload image using Streamlit's file uploader
image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Load the pre-trained Keras model and compile it manually
my_model = models.load_model("model87.h5", compile=False)
my_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Check model input shape
input_shape = my_model.input_shape[1:]
print('Model input shape:', input_shape)

# Function to load the image
def load_image(imageFile):
    img = Image.open(imageFile)
    return img

# Main logic for prediction when an image is uploaded
if image_file is not None:
    # Class labels
    classes = ["non-cancer", "cancer"]
    
    # Display the uploaded image
    st.image(load_image(image_file), width=256)
    
    # Load and preprocess the image
    img = Image.open(image_file)
    img_resized = img.resize((50, 50))
    img_array = np.asarray(img_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    
    # Output the shape of the input image
    print('Input shape:', img_input.shape)
    
    # Predict the class using the model
    result = my_model.predict(img_input)
    ind = np.argmax(result)
    final_output_prediction = classes[ind]
    
    # Display the prediction result
    print(final_output_prediction)
    st.header(final_output_prediction)
