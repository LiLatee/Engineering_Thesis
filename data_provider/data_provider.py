from csvsort import csvsort
import pandas as pd
import asyncio
import websockets
import requests
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

not_sorted_data_file_name = '../data/CriteoSearchData.csv'
data_file_name = 'data/CriteoSearchData-sorted-no-duplicates.csv'
train_model_samples_number = 1000

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


async def consumer_handler(websocket, path) -> None:
    async for message in websocket:
        if message == 'start':
            await process_all_samples(websocket, path)


async def process_all_samples(websocket, path) -> None:
    send_samples_for_model_training()
    chunksize = 3
    samples_num = 0

    for chunk in pd.read_csv(
            data_file_name,
            sep='\t',
            chunksize=chunksize,
            nrows=10000,
            skiprows=(1, train_model_samples_number),
            header=0,
            usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]):
        for index, row in chunk.iterrows():
            samples_num += 1
            # print(samples_num)
            await send_model_info_to_websocket(websocket, samples_num)
            session = requests.Session()
            retry = Retry(connect=3, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            session.post(url="http://prediction_server:5000/predict", data=row.to_json())
            # time.sleep(0.02)

            # requests.request(method='POST', url='http://prediction_server:5000/predict', data=row.to_json())
            # if samples_num % 100 == 0:
            #     requests.request(method='GET', url='http://prediction_server:5000/update_start', data=chunk.to_json())


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
        "model_version": get_model_version()
    }
    # print(message)
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
    # time.sleep(0.5)

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
    # asyncio.get_event_loop().run_until_complete(server_waiting_for_start)
    # asyncio.get_event_loop().run_forever()


    # 1. NAUCZ PIERWSZY MODEL
    import zmq
    context = zmq.Context()
    fit_socket = context.socket(zmq.PAIR)
    fit_socket.connect('tcp://127.0.0.1:5001')

    data = pd.read_csv(
        data_file_name,
        sep='\t',
        nrows=1000,
        header=0,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

    print('sending...')
    fit_socket.send_string(data.to_json(orient='records'))  # convert from bytes to string

    result = fit_socket.recv()  # wait for end of fitting
    print('done' + str(result))


    # 2. WYŚLIJ PRÓBKI DO PREDYKCJI
    import pika
    import time

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='172.18.0.5'))
    channel = connection.channel()
    channel.exchange_declare(exchange='prediction_queue_fanout', exchange_type='fanout')

    print("Sending...")
    start = time.time()
    chunksize = 1
    samples_num = 0
    list_of_times = []
    data = pd.read_csv(
            data_file_name,
            sep='\t',
            nrows=10000,
            skiprows=(1, train_model_samples_number),
            header=0,
            usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    x= None
    for index, row in data.iterrows():
        channel.basic_publish(exchange='prediction_queue_fanout', routing_key='', body=row.to_json())
        # time.sleep(0.01)
        x = index
        time.sleep(0.01)
    print(x)

        # print(" [.] Got %r" % response)
    print(time.time()-start)

    connection.close()