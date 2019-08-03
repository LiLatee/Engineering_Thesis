from csvsort import csvsort
import pandas as pd
import asyncio
import websockets
import requests

not_sorted_data_file = 'data/CriteoSearchData.csv'
data_file = 'data/CriteoSearchDataSorted.csv'

headers = ['Sale', 'SalesAmountInEuro', 'time_delay_for_conversion', 'click_timestamp', 'nb_clicks_1week',
           'product_price', 'product_age_group', 'device_type', 'audience_id', 'product_gender',
           'product_brand', 'product_category(1)', 'product_category(2)', 'product_category(3)',
           'product_category(4)', 'product_category(5)', 'product_category(6)', 'product_category(7)',
           'product_country', 'product_id', 'product_title', 'partner_id', 'user_id']


def sort_data_file_by_timestamp():
    timestamp_column_number = [3]
    csvsort(not_sorted_data_file,
            timestamp_column_number,
            output_filename=data_file,
            has_header=False,
            delimiter='\t')


async def process_all_samples(websocket, path):
    chunksize = 1000
    samples_num = 0
    for chunk in pd.read_csv(
            data_file,
            sep='\t',
            chunksize=chunksize,
            names=headers,
            usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]):
        print(chunk)
        for index, row in chunk.iterrows():
            samples_num += 1
            print(samples_num)
            await websocket.send(str(samples_num))
            requests.request(method='POST', url='http://127.0.0.1:5000/fit', data=row.to_json())


async def consumer_handler(websocket, path):
    async for message in websocket:
        if message == 'start':
            await process_all_samples(websocket, path)


if __name__ == "__main__":
    server_waiting_for_start = websockets.serve(consumer_handler, "127.0.0.1", 8765)

    asyncio.get_event_loop().run_until_complete(server_waiting_for_start)
    asyncio.get_event_loop().run_forever()