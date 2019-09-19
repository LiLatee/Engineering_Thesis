import zmq
import json
from flask import Flask, request
from model_SGDClassifier import ModelSGDClassifier


app = Flask(__name__)
model = ModelSGDClassifier()
counter_to_update_model = 0
counter_to_load_model = 0
context = zmq.Context()
fit_socket = context.socket(zmq.PAIR)
fit_socket.connect("tcp://fit_model_server:5001")
update_socket = context.socket(zmq.PUSH)
update_socket.connect("tcp://update_model_server:5002")

@app.route('/predict', methods=['POST'])
def predict():
    global counter_to_load_model
    global counter_to_update_model

    if counter_to_load_model >= 2:
        model.load_model()
        print("loaded nmodel")
        counter_to_load_model = 0
    if counter_to_update_model >= 5:
        model.update_model()
        print("updated nmodel")
        counter_to_update_model = 0

    #todo wynik dać do kolejki zmq evaluation servera
    result = model.predict(request.data)
    counter_to_update_model += 1
    counter_to_load_model += 1
    return str(result)


# dodatkowe
@app.route('/fit', methods=['POST'])
def fit():
    fit_socket.send_string(json.dumps(json.loads(request.data)))  # convert from bytes to string
    result = fit_socket.recv()  # wait for end of fitting
    return str(result)

@app.route('/update', methods=['GET'])
def update_start():
    update_socket.send_string("update_model")
    return ''


if __name__  == "__main__":
    app.run(host='0.0.0.0', port=5000)
