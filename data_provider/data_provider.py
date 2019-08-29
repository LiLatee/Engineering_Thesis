from csvsort import csvsort
import pandas as pd
import asyncio
import websockets
import requests
from redis_client import get_model_version
import json

not_sorted_data_file_name = '../data/CriteoSearchData.csv'
data_file_name = 'data/CriteoSearchData-sorted-no-duplicates.csv'
train_model_samples_number = 10000

headers = ['Sale', 'SalesAmountInEuro', 'time_delay_for_conversion', 'click_timestamp', 'nb_clicks_1week',
           'product_price', 'product_age_group', 'device_type', 'audience_id', 'product_gender',
           'product_brand', 'product_category(1)', 'product_category(2)', 'product_category(3)',
           'product_category(4)', 'product_category(5)', 'product_category(6)', 'product_category(7)',
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
            skiprows=train_model_samples_number,
            names=headers,
            usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]):
        print(chunk)
        for index, row in chunk.iterrows():
            samples_num += 1
            # print(samples_num)
            await send_model_info_to_websocket(websocket, samples_num)
            requests.request(method='POST', url='http://prediction_server:5000/predict', data=row.to_json())
            if samples_num % 100 == 0:
                requests.request(method='GET', url='http://prediction_server:5000/update_start', data=chunk.to_json())


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

    requests.request(method='POST', url='http://prediction_server:5000/fit', data=data.to_json(orient='records'))
    print('data for training was send. ' + str(data.shape))


if __name__ == "__main__":
    server_waiting_for_start = websockets.serve(consumer_handler, "0.0.0.0", 8765)
    asyncio.get_event_loop().run_until_complete(server_waiting_for_start)
    asyncio.get_event_loop().run_forever()