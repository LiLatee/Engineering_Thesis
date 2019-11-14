from csvsort import csvsort
import pandas as pd
import asyncio
import websockets
import requests
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import threading

start = False

not_sorted_data_file_name = '../data/CriteoSearchData.csv'
data_file_name = 'data/CriteoSearchData-sorted-no-duplicates.csv'
train_model_samples_number = 1000
data_file_chunksize = 3
data_file_nrows = 10000
number_of_threads = 5
generator_lock = threading.Lock()

processed_samples_number = 0

headers = ['sale', 'sales_amount_in_euro', 'time_delay_for_conversion', 'click_timestamp', 'nb_clicks_1week',
           'product_price', 'product_age_group', 'device_type', 'audience_id', 'product_gender',
           'product_brand', 'product_category_1', 'product_category_2', 'product_category_3',
           'product_category_4', 'product_category_5', 'product_category_6', 'product_category_7',
           'product_country', 'product_id', 'product_title', 'partner_id', 'user_id']


def sort_data_file_by_timestamp() -> None:
    timestamp_column_number = [3]
    csvsort(not_sorted_data_file_name,
            timestamp_column_number,
            output_filename=data_file_name,
            has_header=False,
            delimiter='\t')


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()
            # return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def data_generator():
    for chunk in pd.read_csv(
            data_file_name,
            sep='\t',
            chunksize=data_file_chunksize,
            nrows=data_file_nrows,
            skiprows=(1, train_model_samples_number),
            header=0,
            usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]):
        for index, row in chunk.iterrows():
            # with generator_lock:
            yield row


# async def consumer_handler(websocket, path) -> None:
#     global start
#     async for message in websocket:
#         if message == 'start':
#             print('MESSAGE=START')
#             start = True
#             return True

async def consumer_handler(websocket, path) -> None:
    async for message in websocket:
        if message == 'start':
            await process_all_samples(websocket, path)


def thread_function(index, generator, websocket):
# async def thread_function(index, generator):
    global processed_samples_number
    # session = requests.Session()
    print(f"Thread {index} started.")
    print(f"Thread id={threading.current_thread()} started")
    while True:
        try:
            data = next(generator)
            print(f"Thread id={threading.current_thread()}")
            processed_samples_number += 1
            # await send_model_info_to_websocket(websocket, processed_samples_number)
            # retry = Retry(connect=3, backoff_factor=0.5)
            # adapter = HTTPAdapter(max_retries=retry)
            # session.mount('http://', adapter)
            # session.mount('https://', adapter)
            requests.post(url="http://prediction_server:5000/predict", data=data.to_json())
            # print(f"Thread {index} - after sending post request.")
            print(f"Thread id={threading.current_thread()} - after sending post request")
        except StopIteration:
            print(f"Thread {index} - StopIteration exception.")
            break


# async def run_threads(generator):
async def run_threads(generator, websocket):
    tasks = list()
    for index in range(number_of_threads):
        tasks.append(thread_function(index, generator))
        # tasks.append(thread_function(index, generator, websocket))
    await asyncio.wait(tasks)


async def process_all_samples(websocket, path) -> None:
# async def process_all_samples() -> None:
    print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa')

    send_samples_for_model_training()
    chunksize = 3
    samples_num = 0
    threads = list()
    generator = data_generator()
    # with ParallelGenerator(
    #     data_generator(),
    #     max_lookahead=100) as generator:
    for index in range(number_of_threads):
        print(f"Thread index={index}")
        thread = threading.Thread(target=thread_function, args=(index, generator, websocket))
        threads.append(thread)
        print(f"Thread index={index} prepared to start")
        thread.start()
        print(f"Thread index={index} started")

    for index, thread in enumerate(threads):
        thread.join()

    tasks = list()
    # for index in range(number_of_threads):
    #     task = asyncio.ensure_future(thread_function(index, generator, websocket))
    #     tasks.append(task)
    #     print(f"Thread index={index} started")

    # NAJNOWSZE:
    # loop = asyncio.new_event_loop()
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(run_threads(generator))
    # loop.run_until_complete(run_threads(generator, websocket))

    # await asyncio.gather(*tasks)

    #
    # session = requests.Session()
    # generator = data_generator()
    # threads = list()

    # for index in range(number_of_threads):
    #     sample = next(generator)
    #     print(f"Thread index={index}")
    #     thread = threading.Thread(target=lambda a: session.post(url="http://prediction_server:5000/predict", data=sample.to_json()), args=(index, generator, websocket))
    #     threads.append(thread)
    #     print(f"Thread index={index} prepared to start")
    #     thread.start()
    #     print(f"Thread index={index} started")

    #

    # for chunk in pd.read_csv(
    #         data_file_name,
    #         sep='\t',
    #         chunksize=chunksize,
    #         nrows=10000,
    #         skiprows=(1, train_model_samples_number),
    #         header=0,
    #         usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]):
    #     for index, row in chunk.iterrows():
    #         samples_num += 1
    #         # print(samples_num)
    #         await send_model_info_to_websocket(websocket, samples_num)
    #         session = requests.Session()
    #         retry = Retry(connect=3, backoff_factor=0.5)
    #         adapter = HTTPAdapter(max_retries=retry)
    #         session.mount('http://', adapter)
    #         session.mount('https://', adapter)
    #         session.post(url="http://prediction_server:5000/predict", data=row.to_json())

    # for index in range(list(chunk.shape)[0]):
    #     samples_num += 1
    #     row = chunk[index:index + 1].to_json(orient='records')
    #     row = json.loads(row)
    #     row = row[0]
    #     row = json.dumps(row)
    #     await send_model_info_to_websocket(websocket, samples_num)
    #     requests.request(method='POST', url='http://prediction_server:5000/predict', data=row)
    #     if samples_num % 100 == 0:
    #             requests.request(method='GET', url='http://prediction_server:5000/update_start', data=chunk.to_json(orient='records'))


async def send_model_info_to_websocket(websocket, samples_num) -> None:
    message = {
        "samples": samples_num,
        "model_name": 0,
        "model_version": 1,
    }
    await websocket.send(json.dumps(message))


def send_samples_for_model_training() -> None:
    data = pd.read_csv(
        data_file_name,
        sep='\t',
        nrows=train_model_samples_number,
        header=0,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.post(url="http://prediction_server:5000/fit", data=data.to_json(orient='records'))

    # requests.request(method='POST', url='http://prediction_server:5000/fit', data=data.to_json(orient='records'))
    print('data for training was send. ' + str(data.shape))



if __name__ == "__main__":
    # chunksize = 3
    # samples_num = 0
    #
    # for chunk in pd.read_csv(
    #         data_file_name,
    #         sep='\t',
    #     nrows=10,
    #         chunksize=chunksize,
    #         skiprows=(1, train_model_samples_number),
    #         header=0,
    #         usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]):
    #     # print(chunk.to_json(orient='records'))
    #     for index in range(list(chunk.shape)[0]):
    #         samples_num += 1
    #         # print(samples_num)
    #         g = chunk[index:index+1].to_json(orient='records')
    #         g = json.loads(g)
    #         g = g[0]
    #         g = json.dumps(g)
    #         print(g)

    # server_waiting_for_start = websockets.serve(consumer_handler, "0.0.0.0", 8765)
    # await consumer_handler(websockets.serve(consumer_handler, "0.0.0.0", 8765), None)
    # ws = create_connection('ws://0.0.0.0:8765/')  # open socket
    # while True:
    #     mess = ws.recv()  # receive from socket
    #     print(mess)
    # ws = create_connection("wss://0.0.0.0:8765/", sslopt={"check_hostname": False})
    # print("Sending 'Hello, World'...")
    # ws.send("Hello, World")
    # print("Sent")
    # print("Receiving...")
    # result = ws.recv()
    # print("Received '%s'" % result)
    # ws.close()
    # ws.close()  # close socket

    # asyncio.get_event_loop().run_until_complete(wait_for_start())
    # print("START!!!!!!!!!!!!!!!!!!!!!!!")

    server_waiting_for_start = websockets.serve(consumer_handler, "0.0.0.0", 8765)
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(server_waiting_for_start)
    # loop.run_until_complete(process_all_samples())
    # wait_for_start = asyncio.get_event_loop().run_until_complete(server_waiting_for_start)
    # asyncio.get_event_loop().run_forever()
    # print(wait_for_start)

    # while True:
    #     if start is True:
    #         print('Start is true:')
    #         loop.run_until_complete(process_all_samples())



    # print('END OF LOOP')

    asyncio.get_event_loop().run_until_complete(server_waiting_for_start)
    asyncio.get_event_loop().run_forever()
