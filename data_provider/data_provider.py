import asyncio
import websockets
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import threading
import data_generator


train_model_samples_number = 1000
number_of_threads = 2


async def consumer_handler(websocket, path) -> None:
    async for message in websocket:
        if message == 'start':
            await process_all_samples()


async def process_all_samples() -> None:
    send_samples_for_model_training()
    generator = data_generator.data_generator()
    threads = list()
    for index in range(number_of_threads):
        thread = threading.Thread(target=thread_function, args=(index, generator))
        threads.append(thread)
        thread.start()

    for index, thread in enumerate(threads):
        thread.join()
    print("All threads have ended")


def send_samples_for_model_training() -> None:
    data = data_generator.get_train_data()
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.post(url="http://prediction_server:5000/fit", data=data.to_json(orient='records'))
    print('Data for training was send. ' + str(data.shape))


def thread_function(index, generator):
    print(f"Thread id={threading.current_thread()} started")
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    while True:
        try:
            data = next(generator)
            session.post(url="http://prediction_server:5000/predict", data=data.to_json())
            print(f"Thread id={threading.current_thread()} - sent data to /predict")
        except StopIteration as e:
            print(f"Thread {index}: {e}")
            break


if __name__ == "__main__":
    server_waiting_for_start = websockets.serve(consumer_handler, "0.0.0.0", 8765)
    asyncio.get_event_loop().run_until_complete(server_waiting_for_start)
    asyncio.get_event_loop().run_forever()
