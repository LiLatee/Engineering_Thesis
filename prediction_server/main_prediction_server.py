import pika
import requests
import threading
import zmq
from model_SGDClassifier import ModelSGDClassifier
import json
import pickle
from client_redis import DatabaseRedis

NUMBER_OF_MODELS = 5

context = zmq.Context()
# fit_socket = context.socket(zmq.PAIR)
# fit_socket.connect("tcp://fit_model_server:5001")
update_socket = context.socket(zmq.PUSH)
update_socket.connect("tcp://build_and_update_model_server:5002")
list_counter_to_update_model = [0] * (NUMBER_OF_MODELS+1) # liczba równoległych modeli +1


def start_new_model(ModelInfo_object, number_of_samples_before_update):
    model = ModelSGDClassifier(ModelInfo_object)
    print("New model has started")
    print(f"number_of_samples_before_update={number_of_samples_before_update}")

    def callback(ch, method, properties, body):
        global list_counter_to_update_model

        if list_counter_to_update_model[ModelInfo_object.id] >= number_of_samples_before_update:
            print("Updating model has started")
            response = requests.request(method="GET",
                                        url='http://sqlite_api:8764/models/get_id_of_last_specified_model/?model_name=SGDClassifier')
            last_model_id = int(response.content)
            if last_model_id == ModelInfo_object.id:
                update_socket.send_string("update_model")
            list_counter_to_update_model[ModelInfo_object.id] = 0

        sample_with_results = model.predict(sample_json=body)
        redis_DB = DatabaseRedis(model_id=ModelInfo_object.id)
        redis_DB.rpush_sample(json_sample=json.dumps(sample_with_results))

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

    number_of_samples_before_update = 0
    current_number_of_models = 0

    while current_number_of_models != NUMBER_OF_MODELS:
        message = info_receiver.recv_string()  # waits for signal to start new model
        message_json = json.loads(message)
        print(f"Received message {message_json}")
        if "number_of_samples_before_update" in message_json:
            number_of_samples_before_update = message_json.get("number_of_samples_before_update")

        response = requests.request(method='GET', url='http://sqlite_api:8764/models/get_last')
        model_info = pickle.loads(response.content)
        thread = threading.Thread(target=start_new_model, args=(model_info, number_of_samples_before_update))
        thread.start()
        current_number_of_models += 1
