import asyncio
import websockets
import threading
import data_generator
import zmq
import pika
import time
import random


train_model_samples_number = 1000
number_of_threads = 2
number_of_sent_samples = 0


async def consumer_handler(websocket, path) -> None:
    async for message in websocket:
        if message == 'start':
            await process_all_samples()


async def process_all_samples() -> None:
    send_samples_for_model_training()
    generator = data_generator.data_generator()
    threads = list()
    for index in range(number_of_threads):
        thread = threading.Thread(target=thread_function, args=(generator, index))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    print("All threads have ended")


def send_samples_for_model_training() -> None:
    data = data_generator.get_train_data()
    context = zmq.Context()
    fit_socket = context.socket(zmq.PAIR)
    fit_socket.connect('tcp://build_and_update_model_server:5001')
    fit_socket.send_string(data.to_json(orient='records'))  # convert from bytes to string
    result = fit_socket.recv()  # wait for end of fitting
    print('Data for training was send. ' + str(data.shape))


def thread_function(generator, index):
    global number_of_sent_samples
    rand = random.Random()
    print(f"{threading.current_thread()} started")
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
    channel = connection.channel()
    channel.exchange_declare(exchange='prediction_queue_fanout', exchange_type='fanout')
    while True:
        try:
            data = next(generator)
            channel.basic_publish(exchange='prediction_queue_fanout', routing_key='', body=data.to_json())
            # print(f"{threading.current_thread()} sent data to /predict; no={number_of_sent_samples}")
            number_of_sent_samples += 1
            time.sleep(rand.randint(0, 500)/10000)
        except StopIteration as e:
            print(f"{threading.current_thread()}: {e}")
            break
    connection.close()


if __name__ == "__main__":
    server_waiting_for_start = websockets.serve(consumer_handler, "0.0.0.0", 8765)
    asyncio.get_event_loop().run_until_complete(server_waiting_for_start)
    asyncio.get_event_loop().run_forever()
