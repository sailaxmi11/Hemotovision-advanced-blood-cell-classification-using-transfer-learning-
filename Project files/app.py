import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("blood_cell.h5")

# Class labels
classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img_path = "static/" + file.filename
    file.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    result = classes[np.argmax(prediction)]

    return render_template('result.html', prediction=result, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
