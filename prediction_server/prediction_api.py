from flask import Flask, request
import json
from MLModel import MLModel

app = Flask(__name__)
model = MLModel()


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/fit', methods=['POST'])
def fit():
    model.fit(request.data)
    return ''


@app.route('/predict', methods=['POST'])
def predict():
    result = model.predict(request.data)
    print("Result=" + str(result))
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
    app.run(host='0.0.0.0', port=5000)