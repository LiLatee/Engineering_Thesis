import zmq
from model_SGDClassifier import ModelSGDClassifier


model = ModelSGDClassifier()

context = zmq.Context()
info_receiver = context.socket(zmq.PAIR)
info_receiver.bind("tcp://127.0.0.1:5000")
while True:
    data_json = info_receiver.recv_string()  # waits for signal to fit
    model.create_model_and_save(data_json)
    info_receiver.send_string("fitted")