import asyncio
import websockets
import threading
import data_generator
import zmq
import pika
import time
import random
import json
import uuid

number_of_threads = 6


def encoder(obj):
    if isinstance(obj, uuid.UUID):
        # if the obj is uuid, we simply return the value of uuid
        return obj.hex
    return json.JSONEncoder.default(obj)

async def consumer_handler(websocket, path) -> None:
    async for message in websocket:
        message_obj = json.loads(message)
        if message_obj['start']:
            send_data_about_number_of_samples_between_updates(message_obj['samples_model_updates'])
            await process_all_samples(int(message_obj['training_dataset_size']))


def send_data_about_number_of_samples_between_updates(samples_model_updates):
    context = zmq.Context()
    fit_socket = context.socket(zmq.PAIR)
    fit_socket.connect('tcp://build_and_update_model_server:5004')
    fit_socket.send_string(samples_model_updates)


async def process_all_samples(training_dataset_size) -> None:
    send_samples_for_model_training(training_dataset_size)
    generator = data_generator.data_generator()
    threads = list()
    for index in range(number_of_threads):
        thread = threading.Thread(target=thread_function, args=(generator, index))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    print("All threads have ended")


def send_samples_for_model_training(training_dataset_size) -> None:


    data = data_generator.get_train_data(training_dataset_size)

    data['id'] = [uuid.uuid1() for _ in range(len(data.index))]

    context = zmq.Context()
    fit_socket = context.socket(zmq.PAIR)
    fit_socket.connect('tcp://build_and_update_model_server:5001')
    fit_socket.send_string(data.to_json(orient='records', default_handler=encoder))  # convert from bytes to string
    result = fit_socket.recv()  # wait for end of fitting
    print('Data for training was send. ' + str(data.shape))


def thread_function(generator, index):
    rand = random.Random()
    print(f"{threading.current_thread()} started")
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
    channel = connection.channel()
    channel.exchange_declare(exchange='prediction_queue_fanout', exchange_type='fanout')
    while True:
        try:
            data = next(generator)
            channel.basic_publish(exchange='prediction_queue_fanout', routing_key='', body=data.to_json(default_handler=encoder))
            # print(f"{threading.current_thread()} sent data to /predict; no={number_of_sent_samples}")
            time.sleep(rand.randint(0, 500)/10000)
        except StopIteration as e:
            print(f"{threading.current_thread()}: {e}")
            break
    connection.close()


if __name__ == "__main__":
    server_waiting_for_start = websockets.serve(consumer_handler, "0.0.0.0", 8765)
    asyncio.get_event_loop().run_until_complete(server_waiting_for_start)
    asyncio.get_event_loop().run_forever()
