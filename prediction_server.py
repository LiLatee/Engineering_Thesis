from flask import Flask, request
import pandas as pd
import json
from model.MLModel import MLModel

app = Flask(__name__)
model = MLModel()


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/fit', methods=['POST'])
def fit():
    # data = pd.DataFrame(data=json.loads(request.data))
    # print(data)
    model.fit(request.data)
    # model.fit(json.loads(request.data))
    return ''


@app.route('/predict', methods=['POST'])
def predict():
    # sample = pd.Series(data=json.loads(request.data))
    # print(sample)
    model.predict(request.data)
    return ''


# starts updating a model
@app.route('/update_start', methods=['GET'])
def update_start():
    model.update_start(json.loads(request.data))
    return ''


# sends signal to model, that updated model is ready and models should be replaced
@app.route('/update_ready', methods=['GET'])
def update_ready():
    model.update_ready(file_name=request.data)
    return ''


if __name__  == "__main__":
    app.run()