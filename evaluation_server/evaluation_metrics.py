import json
from typing import List


def get_num_of_good_predictions(samples: List[bytes]) -> int:
    predicted = 0
    for sample in samples:
        sample_json = json.loads(sample.decode('utf8'))
        print(sample_json.get("Sale"))
        print(sample_json.get("predicted"))
        if str(sample_json.get("Sale")) == str(sample_json.get("predicted")):
            predicted += 1
    return predicted


def is_prediction_correct(sample: dict) -> bool:
    return str(sample.get("Sale")) == str(sample.get("predicted"))

