from flask import Flask, request
from MLModel import MLModel
from cass_client import CassandraClient
import json

app = Flask(__name__)
model = MLModel()
cass = CassandraClient()

@app.route('/')
def hello_world():
    return str(cass.get_model_history_all())
    # return 'Hello, World!'


@app.route('/restart')
def restart():
    cass.restart_cassandra()
    return 'Cassandra restarted'


@app.route('/sample')
def sample():
    return str(cass.get_all_samples_as_list_of_dicts())
    # return str(len(get_sample_all()))


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