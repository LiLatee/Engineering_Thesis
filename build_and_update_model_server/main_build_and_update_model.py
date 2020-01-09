import zmq
import time
import threading
import json
from model_SGDClassifier import ModelSGDClassifier


number_of_samples_before_update = 0


def get_number_of_samples_before_update():
    global number_of_samples_before_update
    context = zmq.Context()
    info_receiver = context.socket(zmq.PAIR)
    info_receiver.bind("tcp://0.0.0.0:5004")

    while True:
        message = info_receiver.recv()
        number_of_samples_before_update = int(message)


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
        message = {
            "fitted": True,
            "number_of_samples_before_update": number_of_samples_before_update
        }
        update_socket.send_string(json.dumps(message))  # send info to start new model and number_of_samples_before_update
        print(f"Message {message} was sent")
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
        update_socket.send_string(json.dumps({"new_model_built": True}))  # send signal to start new model
        print("Message to start new model was sent")


if __name__ == '__main__':
    get_number_of_samples_before_update_thread = threading.Thread(target=get_number_of_samples_before_update, args=())
    get_number_of_samples_before_update_thread.start()

    building_model_thread = threading.Thread(target=building_model, args=())
    building_model_thread.start()

    updating_model_thread = threading.Thread(target=updating_model, args=())
    updating_model_thread.start()



