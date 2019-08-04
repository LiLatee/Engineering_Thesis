from flask import Flask, request
import pandas as pd
import json
from MLModel import MLModel

app = Flask(__name__)
model = MLModel('Some ML model')


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/train', methods=['POST'])
def train():
    data = pd.DataFrame(data=json.loads(request.data))
    # print(data)
    model.train(data)
    return ''


@app.route('/fit', methods=['POST'])
def fit():
    sample = pd.Series(data=json.loads(request.data))
    # print(sample)
    model.fit(sample)
    return ''


@app.route('/update', methods=['GET'])
def update():
    data = pd.DataFrame(data=json.loads(request.data))
    model.update(data)
    return ''

if __name__  == "__main__":
    app.run()