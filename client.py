from csvsort import csvsort
import pandas as pd
import asyncio
import websockets

not_sorted_data_file = 'data/CriteoSearchData.csv'
data_file = 'data/CriteoSearchDataSorted.csv'


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
    for chunk in pd.read_csv(data_file, chunksize=chunksize):
        for sample in chunk:
            samples_num += 1
            print(samples_num)
            await websocket.send(str(samples_num))


if __name__ == "__main__":
    start_server = websockets.serve(process_all_samples, "127.0.0.1", 8765)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()