from flask import Flask, request
import pandas as pd
import json

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/train', methods=['POST'])
def train():
    data = pd.DataFrame(data=json.loads(request.data))
    print("Training data: ")
    print(data)
    return ''


@app.route('/fit', methods=['POST'])
def fit():
    sample = pd.Series(data=json.loads(request.data))
    print(sample)
    return ''


if __name__  == "__main__":
    app.run()