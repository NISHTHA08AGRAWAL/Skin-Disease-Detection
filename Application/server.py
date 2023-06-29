from flask import Flask, request, render_template
import tensorflow as tf
import pickle
import numpy as np
from PIL import Image

app = Flask(__name__)
cnn_model = tf.keras.models.load_model('balance_model.h5')
svm_model = pickle.load(open('svm.pkl','rb'))
bayes_model = pickle.load(open('bayes.pkl','rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the file and radio button value from the POST request
    choice = request.form.get('choice')
    file = request.files['image']
    labels = ['Melanocytic nevi','Melanoma','Benign keratosis-like lesions ','Basal cell carcinoma','Actinic keratoses','Vascular lesions','Dermatofibroma']
    # Open the image using PIL
    img = Image.open(file)
    
    # Preprocess the image
    img = img.resize((64, 64)) # Resize the image to match the input size of the model
    img = np.array(img) # Convert PIL image to numpy array
    
    # Make a prediction using the appropriate model
    if choice == 'cnn':
        x_mean = np.mean(img)
        x_std = np.std(img)
        img = (img - x_mean) / x_std
        prediction = cnn_model.predict(np.expand_dims(img, axis=0))
    elif choice == 'svm':
        img = img / 255.0 # Normalize the image
        prediction = svm_model.predict(np.expand_dims(img.flatten(), axis=0))
    else :
        img = img / 255.0 # Normalize the image
        prediction = bayes_model.predict(np.expand_dims(img.flatten(), axis=0))

    # Convert the prediction to a human-readable format
    print("Prediction : ",prediction)
    print(prediction[0])
    label = labels[np.argmax(prediction)]
    # Return the prediction as JSON
    prediction = prediction.flatten().round(3)
    
    if(choice == 'cnn'):
        return render_template('index1.html' , prediction_text="Result : "+label , table_heading="Probability of every class :" , d0=labels[0] , d1=labels[1] , d2=labels[2] , d3=labels[3] , d4=labels[4], d5=labels[5], d6=labels[6] , p0=prediction[0] , p1=prediction[1] , p2=prediction[2] , p3=prediction[3] , p4=prediction[4] , p5=prediction[5] , p6=prediction[6])
    else:
        return render_template('index1.html' , prediction_text="Result : "+labels[prediction[0]])
if __name__ == '__main__':
    app.run()