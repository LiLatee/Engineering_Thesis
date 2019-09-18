import zmq

from flask import Flask, request
from model_SGDClassifier import ModelSGDClassifier

app = Flask(__name__)
model = ModelSGDClassifier()
counter_to_update_model = 0
counter_to_load_model = 0
context = zmq.Context()
fit_socket = context.socket(zmq.PAIR).bind("tcp://127.0.0.1:5001")
update_socket = context.socket(zmq.PUSH).bind("tcp://127.0.0.1:5002")

@app.route('/predict', methods=['POST'])
def predict():
    if counter_to_load_model >= 200:
        model.load_model()
        counter_to_load_model = 0
    if counter_to_update_model >= 500:
        model.update_model()
        counter_to_update_model = 0

    #todo wynik dać do kolejki zmq evaluation servera
    model.predict(request.data)
    counter_to_update_model += 1
    counter_to_load_model += 1
    return ''


# dodatkowe
@app.route('/fit', methods=['POST'])
def fit():
    fit_socket.send_json(request.data)
    fit_socket.recv_string()  # wait for end of fitting
    return ''

@app.route('/update', methods=['GET'])
def update_start():
    update_socket.send_string("update_model")
    return ''


if __name__  == "__main__":
    app.run(host='0.0.0.0', port=5000)
