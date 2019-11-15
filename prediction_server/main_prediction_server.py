import pika
import requests
import threading
import zmq
from model_SGDClassifier import ModelSGDClassifier
import json
import time
import pickle

NUMBER_OF_SAMPLES_BEFORE_UPDATE = 300

context = zmq.Context()
# fit_socket = context.socket(zmq.PAIR)
# fit_socket.connect("tcp://fit_model_server:5001")
update_socket = context.socket(zmq.PUSH)
update_socket.connect("tcp://update_model_server:5002")


list_counter_to_update_model = [0] * 9 # liczba równoległych modeli -1

def worker(ModelInfo_object):
    model = ModelSGDClassifier(ModelInfo_object)

    def callback(ch, method, properties, body):
        global list_counter_to_update_model
        # file = open("test.txt", 'a+')
        # try:
        #     file.write(str(list_counter_to_update_model) + '\t')
        #     file.write(str(ModelInfo_object.id) + '\t')
        #     file.write(str(type(ModelInfo_object.id)) + '\n')
        #     file.write(str(list_counter_to_update_model[ModelInfo_object.id]) + '\n')
        # except Exception as e:
        #     file.write("EXCEPTION: " + str(e) + "\n")

        if list_counter_to_update_model[ModelInfo_object.id] >= NUMBER_OF_SAMPLES_BEFORE_UPDATE:
            print("updating model started")
            response = requests.request(method="GET",
                                        url='http://sqlite_api:8764/models/get_id_of_last_specified_model/?model_name=SGDClassifier')
            last_model_id = int(response.content)
            if last_model_id == ModelInfo_object.id:
                update_socket.send_string("update_model")
            list_counter_to_update_model[ModelInfo_object.id] = 0

        model.predict(sample_json=body)
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


if __name__  == "__main__":
    context = zmq.Context()
    info_receiver = context.socket(zmq.PULL)
    info_receiver.bind("tcp://0.0.0.0:5003")  # queue to inform about new model
    while True:
        info_receiver.recv_string()  # waits for signal to start new model
        response = requests.request(method='GET', url='http://sqlite_api:8764/models/get_last')
        model_info = pickle.loads(response.content)
        thread = threading.Thread(target=worker, args=(model_info,))
        thread.start()