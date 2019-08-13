import asyncio
import websockets
import json
import random
import time


async def wait_for_start(websocket, path):
    async for message in websocket:
        if message == 'start':
            await send_current_evaluation_metrics(websocket, path)


async def send_current_evaluation_metrics(websocket, path):
    while True:
        message = {
            'processed_samples': 500,
            'correct_predictions': random.randint(400, 500),

        }
        await websocket.send(json.dumps(message))

        time.sleep(1)


if __name__ == "__main__":
    sending_evaluation_metrics = websockets.serve(wait_for_start, "0.0.0.0", 8766)

    asyncio.get_event_loop().run_until_complete(sending_evaluation_metrics)
    asyncio.get_event_loop().run_forever()