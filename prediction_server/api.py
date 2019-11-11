import pika
import requests
import threading
import zmq
from model_SGDClassifier import ModelSGDClassifier
import json
import time
import pickle

context = zmq.Context()
fit_socket = context.socket(zmq.PAIR)
fit_socket.connect("tcp://fit_model_server:5001")
update_socket = context.socket(zmq.PUSH)
update_socket.connect("tcp://update_model_server:5002")


counter_to_update_model = 0
counter_to_load_model = 0


def worker(ModelInfo_object): # todo wybieranie modelu
    model = ModelSGDClassifier(ModelInfo_object)  # użyć model_id_or_name

    def callback(ch, method, properties, body):
        global counter_to_load_model
        global counter_to_update_model

        if counter_to_load_model >= 100:
            model.load_model_if_exists()
            print("loaded model")
            counter_to_load_model = 0
        if counter_to_update_model >= 200:
            print("updating model started")
            update_socket.send_string("update_model")
            counter_to_update_model = 0

        model.predict(sample_json=body)
        counter_to_update_model = counter_to_update_model + 1
        counter_to_load_model = counter_to_load_model + 1
        ch.basic_ack(delivery_tag=method.delivery_tag)

    connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))  # todo zmienić na kontener
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



# counter_to_update_model = 0
# counter_to_load_model = 0
#
# def on_request(ch, method, props, body):
#     # global counter_to_load_model
#     # global counter_to_update_model
#     #
#     # if counter_to_load_model >= 100:
#     #     model.load_model_if_exists()
#     #     print("loaded nmodel")
#     #     counter_to_load_model = 0
#     # if counter_to_update_model >= 200:
#     #     # model.update_model()
#     #     print("updating model started")
#     #     requests.request(method='GET', url='http://127.0.0.1:5000/update')
#     #     counter_to_update_model = 0
#
#     #todo wynik dać do kolejki zmq evaluation servera
#     response = model.predict(body)
#     # counter_to_update_model += 1
#     # counter_to_load_model += 1
#
#     # response = model.predict(body)
#
#     ch.basic_publish(exchange='',
#                      routing_key=props.reply_to,
#                      properties=pika.BasicProperties(correlation_id=props.correlation_id),
#                      body=str(response))
#     ch.basic_ack(delivery_tag=method.delivery_tag)
#
# channel.basic_qos(prefetch_count=1)
# channel.basic_consume(queue='prediction_queue', on_message_callback=on_request)


if __name__  == "__main__":
    pass
    # channel.start_consuming()

    #start all models in db
    # response = requests.request(method="GET",
    #                             url='http://sqlite_api:8764/models/get_as_list_of_ModelInfo')
    # all_models = pickle.loads(response.content)
    # for model in all_models:
    #     thread = threading.Thread(target=worker, args=(model,))
    #     thread.start()

    # start first model
    response = requests.request(method='GET', url='http://sqlite_api:8764/models/get_last')
    model_info = pickle.loads(response.content)
    model = ModelSGDClassifier(model_info)
    thread = threading.Thread(target=worker, args=(model,))
    thread.start()

    context = zmq.Context()
    info_receiver = context.socket(zmq.PULL)
    info_receiver.bind("tcp://0.0.0.0:5003")
    while True:
        info_receiver.recv_string()  # waits for signal to start new model
        file = open("test.txt", "a+")
        file.write("omg")
        file.close()
        response = requests.request(method='GET', url='http://sqlite_api:8764/models/get_last')
        model_info = pickle.loads(response.content)
        model = ModelSGDClassifier(model_info)
        thread = threading.Thread(target=worker, args=(model,))
        thread.start()