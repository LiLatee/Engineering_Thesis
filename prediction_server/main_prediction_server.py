import pika
import requests
import threading
import zmq
from model_SGDClassifier import ModelSGDClassifier
import json
import time
import pickle

NUMBER_OF_SAMPLES_BEFORE_UPDATE = 2500
NUMBER_OF_MODELS = 8

context = zmq.Context()
# fit_socket = context.socket(zmq.PAIR)
# fit_socket.connect("tcp://fit_model_server:5001")
update_socket = context.socket(zmq.PUSH)
update_socket.connect("tcp://build_and_update_model_server:5002")
list_counter_to_update_model = [0] * (NUMBER_OF_MODELS+1) # liczba równoległych modeli +1


def start_new_model(ModelInfo_object):
    model = ModelSGDClassifier(ModelInfo_object)
    print("New model started")

    def callback(ch, method, properties, body):
        global list_counter_to_update_model

        if list_counter_to_update_model[ModelInfo_object.id] >= NUMBER_OF_SAMPLES_BEFORE_UPDATE:
            print("Updating model started")
            response = requests.request(method="GET",
                                        url='http://sqlite_api:8764/models/get_id_of_last_specified_model/?model_name=SGDClassifier')
            last_model_id = int(response.content)
            if last_model_id == ModelInfo_object.id:
                update_socket.send_string("update_model")
            list_counter_to_update_model[ModelInfo_object.id] = 0

        model.predict(sample_json=body)
        # print("Prediction was made")
        list_counter_to_update_model[ModelInfo_object.id] = list_counter_to_update_model[ModelInfo_object.id] + 1
        ch.basic_ack(delivery_tag=method.delivery_tag)

    connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
    channel = connection.channel()
    channel.exchange_declare(exchange='prediction_queue_fanout', exchange_type='fanout')
    channel.basic_qos(prefetch_count=1)
    queue = channel.queue_declare(queue='', exclusive=True)
    queue_name = queue.method.queue
    channel.queue_bind(exchange='prediction_queue_fanout', queue=queue_name)
    channel.basic_consume(
        queue=queue_name,
        on_message_callback=callback,
        # auto_ack=True
    )
    channel.start_consuming()


if __name__ == "__main__":
    context = zmq.Context()
    info_receiver = context.socket(zmq.PULL)
    info_receiver.bind("tcp://0.0.0.0:5003")  # queue to inform about new model

    current_number_of_models = 0
    while current_number_of_models != NUMBER_OF_MODELS:
        info_receiver.recv_string()  # waits for signal to start new model
        response = requests.request(method='GET', url='http://sqlite_api:8764/models/get_last')
        model_info = pickle.loads(response.content)
        thread = threading.Thread(target=start_new_model, args=(model_info,))
        thread.start()
        current_number_of_models = current_number_of_models + 1


