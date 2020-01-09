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


def get_roc_auc_score(samples: List) -> float:
    y_true = [int(float(sample_bytes_to_json(sample).get("sale"))) for sample in samples]
    # y_scores = [float(sample_bytes_to_json(sample).get("probabilities")[6:9]) for sample in samples]
    y_scores = [float(json.loads(sample_bytes_to_json(sample).get("probabilities"))[1]) for sample in samples]

    # print('y_true:', y_true, flush=True)
    # print('y_true:', len(y_true), flush=True)
    # print('y_scores', y_scores, flush=True)
    # print('y_scores', len(y_scores), flush=True)
    # return roc_auc_score(y_scores, y_true)
    # tn, fp, fn, tp = confusion_matrix(y_true, y_scores).ravel()
    # print(f"confusion matrix tn, fp, fn, tp = {tn, fp, fn, tp}")


    # file2 = open('test2.txt', 'a+')
    # file2.write(str(sample_bytes_to_json(samples[3]).get("probabilities")[6:9]))
    # file2.write('\n')
    # file2.close()
    #
    # file = open('test.txt','a+')
    # file.write(str(y_true))
    # file.write('\n')
    # file.write(str(y_scores))
    # file.write('\n')
    # file.write('\n')
    # file.close()
    return roc_auc_score(y_true, y_scores)


def get_f1_score(samples: List) -> float:
    y_true = [int(float(sample_bytes_to_json(sample).get("sale"))) for sample in samples]
    y_pred = [int(float(sample_bytes_to_json(sample).get("predicted"))) for sample in samples]

    return f1_score(y_true, y_pred)


def sample_bytes_to_json(sample: bytes) -> Union[str, Any]:
    return json.loads(sample.decode('utf8'))
