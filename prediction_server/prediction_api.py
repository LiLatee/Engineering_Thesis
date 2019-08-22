from flask import Flask, request
import json
from MLModel import MLModel
from cass_client import create_tables, get_model_history_all, add_some_data, get_sample_all
import json

app = Flask(__name__)
model = MLModel()

@app.route('/')
def hello_world():
    return str(get_model_history_all())
    # return 'Hello, World!'


@app.route('/data')
def data():
    add_some_data()
    return 'Data added properly'


@app.route('/sample')
def sample():
    return str(get_sample_all())


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
    model.update_ready()
    return ''


if __name__  == "__main__":
    app.run(host='0.0.0.0', port=5000)