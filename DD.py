#from flask import Flask , request , render_template ,url_for,jsonify
from tensorflow.keras.models import load_model
#from PIL import Image
import numpy as np
#from tensorflow import keras
import streamlit as st
import cv2
def preprossing(image):
     Image=cv2.resize(image,(50,50))
     Image=Image.astype("float32")/255.
     Image=Image.reshape((1, ) +Image.shape)
     return Image

st.title(" Oral Cancer Detection ")
image_file= st.file_uploader("image upload",type=["png","jpg","jpeg"])
my_model= load_model("D:\DMC Project\github\oral_cancer\model87.h5")
input_shape = my_model.layers[0].input_shape[1:]
print('Input shape---------------------------------------:', input_shape)
def load_image(imageFile):
    img= Image.open(imageFile)
    return img
if image_file is not None:
     classes = ["non-cancer","cancer"]
     st.image(load_image(image_file),width=256)
    # Load image file
     img = Image.open(image_file)
# Resize image to 50x50 pixels
     img_resized = img.resize((50, 50))
# Convert image to numpy array and normalize pixel values
     img_array = np.asarray(img_resized) / 255.0
# Add an extra dimension to the image
     img_input = np.expand_dims(img_array, axis=0)

# Print the shape of the input image
     print('Input shape:', img_input.shape)
     result= my_model.predict(img_input)
     ind = np.argmax(result)
     final_output_prediction= classes[ind]
     print("Amr an shaa allaha will be the best")
     print(final_output_prediction)
     st.header(final_output_prediction)



print("ya raaaaaaaaab")
#rom tensorflow.keras.models import load_model
#print(tf.__version__)
# my_model.predict()
# app=Flask(__name__)


# @app.route('/')
# def index():
#      return render_template("index.html")


# @app.route('/predict', methods=["GET","POST"])
# def predict():
#      print("run cooode")
#      if request.method=="POST":
#           print ("Trying load image correctly")
#           image = request.files["fileup"]
#           print ("loaded image correctly")
#           image = preprossing(image)
#           print("image is preprossed correctly")
#           result=[my_model.predict(image)]
#           ind = np.argmax(result)
#           final_output_prediction= classes[ind]
#           print("the model worked well ")
     
#      return final_output_prediction