import asyncio
import websockets
import json
import time
from redis_client import RedisClient
from cass_client import CassandraClient
from evaluation_metrics import is_prediction_correct, get_roc_auc_score
from typing import Union, Any


class EvaluationServer:

    def __init__(self) -> None:
        self.redis = RedisClient()
        self.cass = CassandraClient()
        self.num_processed_samples = 0
        self.correct_predictions = 0

    async def wait_for_start(self, websocket, path) -> None:
        async for message in websocket:
            if message == 'start':
                await self.send_current_evaluation_metrics(websocket)

    async def send_current_evaluation_metrics(self, websocket) -> None:
        while True:
            message = self.process_latest_samples_and_create_message()
            await websocket.send(json.dumps(message))
            time.sleep(1)

    def process_latest_samples_and_create_message(self) -> dict:
        processed_samples = self.redis.get_all_samples_as_list_of_bytes()
        for sample in processed_samples:
            sample_json = json.loads(sample.decode('utf8'))
            self.num_processed_samples += 1
            self.check_correct_prediction(sample_json)
            self.cass.insert_sample(sample_json)
        roc_auc_score = 0
        if self.num_processed_samples > 50:
            roc_auc_score = self.calculate_roc_auc_score(None, None)
        return {
            "processed_samples": self.num_processed_samples,
            "correct_predictions": self.correct_predictions,
            "roc_auc_score": roc_auc_score
        }

    def check_correct_prediction(self, sample: Union[str, Any]) -> None:
        if is_prediction_correct(sample):
            self.correct_predictions += 1

    def calculate_roc_auc_score(self, id_first_sample: int, id_last_sample: int):
        samples = self.cass.get_all_samples_as_list_of_dicts()
        return get_roc_auc_score(samples)


if __name__ == "__main__":
    evaluation_server = EvaluationServer()
    sending_evaluation_metrics = websockets.serve(evaluation_server.wait_for_start, "0.0.0.0", 8766)
    asyncio.get_event_loop().run_until_complete(sending_evaluation_metrics)
    asyncio.get_event_loop().run_forever()