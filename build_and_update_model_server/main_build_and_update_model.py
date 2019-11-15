import zmq
import time
import threading
from model_SGDClassifier import ModelSGDClassifier

def building_model():
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


def updating_model():
    model = ModelSGDClassifier()
    context = zmq.Context()
    info_receiver = context.socket(zmq.PULL)
    info_receiver.bind("tcp://0.0.0.0:5002")

    update_socket = context.socket(zmq.PUSH)
    update_socket.connect("tcp://prediction_server:5003")
    while True:
        info_receiver.recv_string()  # waits for signal to update
        model.load_last_model()
        model.update_model()
        update_socket.send_string("new_model_built")  # send signal to start new model

if __name__ == '__main__':
    building_model_thread = threading.Thread(target=building_model, args=())
    building_model_thread.start()

    updating_model_thread = threading.Thread(target=updating_model, args=())
    updating_model_thread.start()



