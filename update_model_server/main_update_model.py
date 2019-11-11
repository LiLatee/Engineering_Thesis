import zmq
from model_SGDClassifier import ModelSGDClassifier


model = ModelSGDClassifier()

context = zmq.Context()
info_receiver = context.socket(zmq.PULL)
info_receiver.bind("tcp://0.0.0.0:5002")

update_socket = context.socket(zmq.PUSH)
update_socket.connect("tcp://prediction_server:5003")
while True:
    info_receiver.recv_string()  # waits for signal to update
    model.load_model_if_exists()
    model.update_model()
    update_socket.send_string("new_model_built")  # send signal to start new model
