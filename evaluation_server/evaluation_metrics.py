import json
from typing import List, Union, Any
from sklearn.metrics import roc_auc_score


def get_num_of_good_predictions(samples: List[bytes]) -> int:
    predicted = 0
    for sample in samples:
        sample_json = sample_bytes_to_json(sample)
        print(sample_json.get("Sale"))
        print(sample_json.get("predicted"))
        if str(sample_json.get("Sale")) == str(sample_json.get("predicted")):
            predicted += 1
    return predicted


def is_prediction_correct(sample: dict) -> bool:
    return str(sample.get("Sale")) == str(sample.get("predicted"))


def get_roc_auc_score(samples: dict) -> float:
    y_true = [sample.get("sale") for sample in samples]
    y_scores = [sample.get("probabilities")[1] for sample in samples]
    # print('y_true:', len(y_true), flush=True)
    # print('y_scores', len(y_scores), flush=True)
    return roc_auc_score(y_true, y_scores)


def sample_bytes_to_json(sample: bytes) -> Union[str, Any]:
    return json.loads(sample.decode('utf8'))
