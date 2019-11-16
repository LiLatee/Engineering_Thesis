import asyncio
import websockets
import json
import time
from client_redis import DatabaseRedis
from client_SQLite import DatabaseSQLite
from typing import Union, Any
from metrics import is_prediction_correct, get_roc_auc_score


class EvaluationServer:

    def __init__(self) -> None:
        self.all_redis_connections = [DatabaseRedis(i) for i in range(1, 8)]
        self.db = DatabaseSQLite()

    async def wait_for_start(self, websocket, path) -> None:
        async for message in websocket:
            if message == 'start':
                await self.send_current_evaluation_metrics(websocket)

    async def send_current_evaluation_metrics(self, websocket) -> None:
        while True:
            message = self.process_latest_samples_and_create_message()
            await websocket.send(json.dumps(message))

    def process_latest_samples_and_create_message(self) -> dict:
        message = {}
        for model in self.all_redis_connections:
            if model.key_exists():
                processed_samples = model.get_all_samples_as_list_of_bytes()
                for sample in processed_samples:
                    sample_json = json.loads(sample.decode('utf8'))
                    model.num_processed_samples += 1
                    self.check_correct_prediction(model, sample_json)
                    self.db.insert_sample_as_dict(sample_json)
                roc_auc_score = 0
                if model.num_processed_samples > 50:
                    roc_auc_score = self.calculate_roc_auc_score(None, None)
                message[model.model_id] = {
                    "processed_samples": model.num_processed_samples,
                    "correct_predictions": model.correct_predictions,
                    "roc_auc_score": roc_auc_score
                }
            else:
                message[model.model_id] = 0
                time.sleep(0.2)
        return message

    def check_correct_prediction(self, model, sample: Union[str, Any]) -> None:
        if is_prediction_correct(sample):
            model.correct_predictions += 1

    def calculate_roc_auc_score(self, id_first_sample: int, id_last_sample: int):
        samples = self.db.get_all_samples_as_list_of_dicts()
        return get_roc_auc_score(samples)


if __name__ == "__main__":
    evaluation_server = EvaluationServer()
    sending_evaluation_metrics = websockets.serve(evaluation_server.wait_for_start, "0.0.0.0", 8766)
    asyncio.get_event_loop().run_until_complete(sending_evaluation_metrics)
    asyncio.get_event_loop().run_forever()
