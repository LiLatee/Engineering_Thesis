import zmq
from model_SGDClassifier import ModelSGDClassifier


model = ModelSGDClassifier()

context = zmq.Context()
info_receiver = context.socket(zmq.PULL)
info_receiver.bind("tcp://127.0.0.1:5000")
while True:
    info_receiver.recv_string()  # waits for signal to update
    model.load_model()
    model.update_model()
