import zmq
from model_SGDClassifier import ModelSGDClassifier
import time
model = ModelSGDClassifier()

context = zmq.Context()
info_receiver = context.socket(zmq.PAIR)
info_receiver.bind("tcp://0.0.0.0:5001")

update_socket = context.socket(zmq.PUSH)
update_socket.connect("tcp://prediction_server:5003")
while True:
    data_json = info_receiver.recv()  # waits for signal to fit
    model.create_model_and_save(data_json)
    update_socket.send_string("fitted")  # send info to start new model
    time.sleep(3)
    info_receiver.send_string("fitted")  # model has been built
