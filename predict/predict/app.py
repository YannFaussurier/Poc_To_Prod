import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request
from run import TextPredictionModel
import json

# model load and cleaning
# model = tf.keras.models.load_model("model.h5")

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def get_text():
    model = TextPredictionModel.from_artefacts("/Users/rvie/Documents/5A/Poc to prod/poc-to-prod-romain/train/data/artefacts/2023-01-03-12-42-59")

    body = json.loads(request.get_data())

    text = body['text']
    top_k = body['top_k']

    predictions = model.predict(text, top_k)

    return "The text is :  " + text + "  and the prediction is :  " + str(predictions)


@app.route('/', methods=['GET'])
def index():
    return 'Flask application for the from poc to prod application, go to /predict to see more'


if __name__ == '__main__':
    app.run(debug=True)