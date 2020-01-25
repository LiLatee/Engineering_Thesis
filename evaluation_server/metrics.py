import json
from typing import List, Union, Any
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


def get_num_of_good_predictions(samples: List[bytes]) -> int:
    predicted = 0
    for sample in samples:
        sample_json = sample_bytes_to_json(sample)
        print(sample_json.get("sale"))
        print(sample_json.get("predicted"))
        if str(sample_json.get("sale")) == str(sample_json.get("predicted")):
            predicted += 1
    return predicted


def is_prediction_correct(sample: dict) -> bool:
    return int(sample.get("sale")) == int(sample.get("predicted"))


def get_roc_auc_score(samples: List,  model_number) -> float:
    y_true = [int(float(sample_bytes_to_json(sample).get("sale"))) for sample in samples]
    y_scores = [float(json.loads(sample_bytes_to_json(sample).get("probabilities"))[1]) for sample in samples]

    # with open('model_' + str(model_number) + "_aucroc-true.txt", 'a+') as f:
    #     f.write(json.dumps(y_true)+'\n')
    # with open('model_' + str(model_number) + "_aucroc-scores.txt", 'a+') as f:
    #     f.write(json.dumps(y_scores)+'\n')

    return roc_auc_score(y_true, y_scores)


def get_f1_score(samples: List,model_number) -> float:
    y_true = [int(float(sample_bytes_to_json(sample).get("sale"))) for sample in samples]
    y_pred = [int(float(sample_bytes_to_json(sample).get("predicted"))) for sample in samples]

    # with open('model_' + str(model_number) + "_f1-true.txt", 'a+') as f:
    #     f.write(json.dumps(y_true)+'\n')
    # with open('model_' + str(model_number) + "_f1-pred.txt", 'a+') as f:
    #     f.write(json.dumps(y_pred)+'\n')

    return f1_score(y_true, y_pred)


def sample_bytes_to_json(sample: bytes) -> Union[str, Any]:
    return json.loads(sample.decode('utf8'))
