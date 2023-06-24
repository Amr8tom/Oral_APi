from flask import Flask , request , render_template ,url_for,jsonify
from tensorflow.keras.models import load_model
from ptl import image
import numpy as np
from tensorflow import keras
print("ya raaaaaaaaab")
#rom tensorflow.keras.models import load_model
my_model= load_model("D:\DMC Project\github\oral_cancer\model87.h5")
#print(tf.__version__)
my_model.predict()
app=Flask(__name__)
def preprossing(image):
     Image=cv2.resize(image,(50,50))
     Image=Image.astype("float32")/255.
     Image=Image.reshape((1, ) +Image.shape)
     return Image


classes = ["non-cancer","cancer"]

@app.route('/')
def index():
     return render_template("index.html")


@app.route('/predict', methods=["GET","POST"])
def predict():
     print("run cooode")
     if request.method=="POST":
          print ("Trying load image correctly")
          image = request.files["fileup"]
          print ("loaded image correctly")
          image = preprossing(image)
          print("image is preprossed correctly")
          result=[my_model.predict(image)]
          ind = np.argmax(result)
          final_output_prediction= classes[ind]
          print("the model worked well ")
     
     return final_output_prediction