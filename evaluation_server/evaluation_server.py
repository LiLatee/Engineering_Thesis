import asyncio
import websockets
import json
import random
import time
# from db_utils.SQLite_client import DatabaseSQLite as db

async def wait_for_start(websocket, path):
    async for message in websocket:
        if message == 'start':
            await send_current_evaluation_metrics(websocket, path)


async def send_current_evaluation_metrics(websocket, path):
    processed_samples = 100
    last_id = 0
    while True:
        processed_samples += 100
        # try:
        #     last_id = db.get_last_sample_id()
        # except:
        #     last_id = 0
        message = {
            'last_sample_id': last_id,
            'processed_samples': processed_samples,
            'correct_predictions': random.randint(processed_samples*0.8, processed_samples),
        }
        await websocket.send(json.dumps(message))

        time.sleep(1)
        last_id = db.get_last_sample_id()


if __name__ == "__main__":
    sending_evaluation_metrics = websockets.serve(wait_for_start, "0.0.0.0", 8766)

    asyncio.get_event_loop().run_until_complete(sending_evaluation_metrics)
    asyncio.get_event_loop().run_forever()