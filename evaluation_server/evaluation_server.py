import asyncio
import websockets
import json
import random
import time
from cass_client import get_sample_all


async def wait_for_start(websocket, path):
    async for message in websocket:
        if message == 'start':
            await send_current_evaluation_metrics(websocket, path)


async def send_current_evaluation_metrics(websocket, path):
    last_id = 0
    while True:
        processed_samples = len(get_sample_all())
        correct_predictions = 0
        if processed_samples > 10:
            correct_predictions = random.randint(int(processed_samples*0.8), processed_samples)
        message = {
            'last_sample_id': last_id,
            'processed_samples': processed_samples,
            'correct_predictions': correct_predictions,
        }
        await websocket.send(json.dumps(message))

        time.sleep(1)


if __name__ == "__main__":
    sending_evaluation_metrics = websockets.serve(wait_for_start, "0.0.0.0", 8766)

    asyncio.get_event_loop().run_until_complete(sending_evaluation_metrics)
    asyncio.get_event_loop().run_forever()