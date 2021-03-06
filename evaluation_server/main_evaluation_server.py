import asyncio
import websockets
import json
import time
from client_cass import CassandraClient
from client_redis import DatabaseRedis
from typing import Union, Any
from metrics import is_prediction_correct, get_roc_auc_score, get_f1_score
import threading
import uuid

class EvaluationServer:

    def __init__(self) -> None:
        self.all_redis_connections = [DatabaseRedis(i) for i in range(1, 9)]
        self.all_cass_connections_for_each_model = [CassandraClient(table_name='model_' + str(i)) for i in range(1, 9)]
        self.cass_connection_for_all_stored_samples = CassandraClient(table_name='all_stored_samples')
        self.models_first_sample_numbers = [-1 for _ in range(0, 8)]
        self.models_first_sample_numbers[0] = 0

    async def wait_for_start(self, websocket, path) -> None:
        async for message in websocket:
            if message == 'start':
                await self.send_current_evaluation_metrics(websocket)

    async def send_current_evaluation_metrics(self, websocket) -> None:
        while True:
            message = self.process_latest_samples_and_create_message()
            if len(message) is not 0:
                # print(message)
                await websocket.send(json.dumps(message))

    def process_latest_samples_and_create_message(self) -> list:
        message = []
        number_od_models = sum([model.key_exists() for model in self.all_redis_connections])
        if number_od_models > 0:
            threads = []
            thread_first_model = threading.Thread(target=self.process_first_model,
                                                  args=(self.all_redis_connections[0], message))
            threads.append(thread_first_model)
            thread_first_model.start()

            for index in range(1, number_od_models):
                thread = threading.Thread(target=self.process_next_models,
                                          args=(self.all_redis_connections[index], message))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
        time.sleep(5)
        return message

    def process_first_model(self, model, message):
        # print(f"{threading.current_thread()} with model {model.model_id} started. "
        #       f"Processed samples = {model.num_processed_samples}")
        processed_samples = model.get_all_samples_as_list_of_json()
        for sample in processed_samples:
            sample_json = json.loads(sample.decode('utf8'))
            del sample_json['predicted']
            del sample_json['probabilities']
            self.cass_connection_for_all_stored_samples.insert_sample(sample_json)
        self.evaluate_model(model, message, processed_samples)

    def process_next_models(self, model, message):
        processed_samples = model.get_all_samples_as_list_of_json()
        self.evaluate_model(model, message, processed_samples)

    def evaluate_model(self, model, message, processed_samples):
        evaluated_number_of_samples = int(self.all_cass_connections_for_each_model[model.model_id - 1].get_number_of_samples_before_id(
            id=uuid.UUID(json.loads(processed_samples[-1])['id']))) - 10000 # todo minus zbiór treningowy
        for sample in processed_samples:
            sample_dict = json.loads(sample.decode('utf8'))
            model.num_processed_samples += 1
            self.check_correct_prediction(model, sample_dict)
            self.all_cass_connections_for_each_model[model.model_id - 1].insert_sample(sample_dict)
        roc_auc_score = 0
        f1_score = 0
        if model.num_processed_samples > 50:
            roc_auc_score = self.calculate_roc_auc_score(processed_samples, model.model_id)
            f1_score = self.calculate_f1_score(processed_samples, model.model_id)


        model.avg_acc = (model.avg_acc*model.num_of_evaluations + model.correct_predictions/model.num_processed_samples)/(model.num_of_evaluations + 1)
        model.avg_aucroc = (model.avg_aucroc*model.num_of_evaluations + roc_auc_score)/(model.num_of_evaluations + 1)
        model.avg_f1_score = (model.avg_f1_score*model.num_of_evaluations + f1_score)/(model.num_of_evaluations + 1)
        model.num_of_evaluations += 1
        print("WYNIKI EWALUACJI:")
        print("model numer: " + str(model.model_id))
        print("Accuracy: " + str(model.avg_acc))
        print("AUC ROC: " + str(model.avg_aucroc))
        print("F1 score: " + str(model.avg_f1_score) + '\n')
        message.append({

            "id": model.model_id,
            "processed_samples": model.num_processed_samples,
            "correct_predictions": model.correct_predictions,
            "roc_auc_score": roc_auc_score,
            "f1_score": f1_score,
            "first_processed_sample": evaluated_number_of_samples
        })

    def get_models_first_sample_number(self, model, first_sample):
        if self.models_first_sample_numbers[model.model_id - 1] is not -1:
            return self.models_first_sample_numbers[model.model_id - 1]
        number_of_sample = self.all_cass_connections_for_each_model[model.model_id - 1].get_number_of_samples_before_id(id=uuid.UUID(json.loads(first_sample.decode('utf8')).get('id')))
        self.models_first_sample_numbers[model.model_id - 1] = number_of_sample
        # print(f"MODEL {model.model_id}, number of first sample = {number_of_sample}")
        return number_of_sample

    def check_correct_prediction(self, model, sample: Union[str, Any]) -> None:
        if is_prediction_correct(sample):
            model.correct_predictions += 1

    def calculate_roc_auc_score(self, samples, model_number):
        return get_roc_auc_score(samples, model_number)

    def calculate_f1_score(self, samples, model_number):
        return get_f1_score(samples, model_number)


if __name__ == "__main__":
    evaluation_server = EvaluationServer()
    sending_evaluation_metrics = websockets.serve(evaluation_server.wait_for_start, "0.0.0.0", 8766)
    asyncio.get_event_loop().run_until_complete(sending_evaluation_metrics)
    asyncio.get_event_loop().run_forever()
