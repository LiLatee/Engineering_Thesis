import asyncio
import websockets
import requests
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
        thread = threading.Thread(target=thread_function, args=(generator, index))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    print("All threads have ended")


def send_samples_for_model_training() -> None:
    data = data_generator.get_train_data()
    requests.post(url="http://prediction_server:5000/fit", data=data.to_json(orient='records'))
    print('Data for training was send. ' + str(data.shape))


def thread_function(generator, index):
    print(f"{threading.current_thread()} started")
    while True:
        try:
            data = next(generator)
            requests.post(url="http://prediction_server:5000/predict", data=data.to_json())
            print(f"{threading.current_thread()} sent data to /predict")
        except StopIteration as e:
            print(f"{threading.current_thread()}: {e}")
            break


if __name__ == "__main__":
    server_waiting_for_start = websockets.serve(consumer_handler, "0.0.0.0", 8765)
    asyncio.get_event_loop().run_until_complete(server_waiting_for_start)
    asyncio.get_event_loop().run_forever()
