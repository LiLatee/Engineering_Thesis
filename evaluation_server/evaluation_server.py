import asyncio
import websockets
import json
import time
from DatabaseRedis import DatabaseRedis
from cass_client import CassandraClient
from evaluation_metrics import is_prediction_correct

class EvaluationServer:

    def __init__(self) -> None:
        self.redis = DatabaseRedis()
        self.cass = CassandraClient()
        self.num_processed_samples = 0
        self.correct_predictions = 0

    async def wait_for_start(self, websocket, path):
        async for message in websocket:
            if message == 'start':
                await self.send_current_evaluation_metrics(websocket, path)

    async def send_current_evaluation_metrics(self, websocket, path):
        while True:
            message = self.process_latest_samples_and_create_message()
            await websocket.send(json.dumps(message))
            time.sleep(1)

    def process_latest_samples_and_create_message(self):
        processed_samples = self.redis.get_all_samples_as_list_of_bytes()
        for sample in processed_samples:
            sample_json = json.loads(sample.decode('utf8'))
            self.num_processed_samples += 1
            self.check_correct_prediction(sample_json)
            self.cass.insert_sample(sample_json)
        return {
            'processed_samples': self.num_processed_samples,
            'correct_predictions': self.correct_predictions,
        }

    def check_correct_prediction(self, sample):
        if is_prediction_correct(sample):
            self.correct_predictions += 1

if __name__ == "__main__":
    evaluation_server = EvaluationServer()
    sending_evaluation_metrics = websockets.serve(evaluation_server.wait_for_start, "0.0.0.0", 8766)

    asyncio.get_event_loop().run_until_complete(sending_evaluation_metrics)
    asyncio.get_event_loop().run_forever()