import os
from flask import Flask, request, redirect, url_for, jsonify, Response
from werkzeug.utils import secure_filename

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)
model = None
graph = tf.Graph()

def load_model():
    global model
    global graph
    with graph.as_default():
        model = VGG16(weights='imagenet', include_top=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    response = {'success': False}
    if request.method == 'POST':
        if request.files.get('file'):
            img_requested = request.files['file'].read()
            img = Image.open(io.BytesIO(img_requested))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            inputs = preprocess_input(img)
            
            with graph.as_default():
                preds = model.predict(inputs)
            results = decode_predictions(preds)

            response['predictions'] = []
            for (imagenetID, label, prob) in results[0]:
                row = {'label': label, 'probability': float(prob)}
                response['predictions'].append(row)
            response['success'] = True
            return jsonify(response)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    load_model()
    # no-thread: https://github.com/keras-team/keras/issues/2397#issuecomment-377914683
    # avoid model.predict runs before model initiated
    app.run(threaded=False)
