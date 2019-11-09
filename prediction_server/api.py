import pika
import requests
from model_SGDClassifier import ModelSGDClassifier

connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq')) #todo zmienić na kontener
channel = connection.channel()
channel.queue_declare(queue='prediction_queue')

model = ModelSGDClassifier()
counter_to_update_model = 0
counter_to_load_model = 0

def on_request(ch, method, props, body):
    # global counter_to_load_model
    # global counter_to_update_model
    #
    # if counter_to_load_model >= 100:
    #     model.load_model_if_exists()
    #     print("loaded nmodel")
    #     counter_to_load_model = 0
    # if counter_to_update_model >= 200:
    #     # model.update_model()
    #     print("updating model started")
    #     requests.request(method='GET', url='http://127.0.0.1:5000/update')
    #     counter_to_update_model = 0

    #todo wynik dać do kolejki zmq evaluation servera
    response = model.predict(body)
    # counter_to_update_model += 1
    # counter_to_load_model += 1

    # response = model.predict(body)

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=str(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='prediction_queue', on_message_callback=on_request)


if __name__  == "__main__":
    channel.start_consuming()