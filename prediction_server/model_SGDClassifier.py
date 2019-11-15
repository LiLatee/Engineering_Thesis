import numpy as np
import json

import data_preprocessing as dp
# from client_SQLite import DatabaseSQLite
from model_info import ModelInfo
from client_redis import DatabaseRedis

from typing import List, Dict, Union, Any, Tuple
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

# JSONType = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONType = Union[str, bytes, bytearray]
RowAsDictType = Dict[str, Union[str, float, int]]


class ModelSGDClassifier:

    def __init__(self, model_info: ModelInfo = None) -> None:
        self.ModelInfo: ModelInfo = model_info
        self.model: SGDClassifier = model_info.model
        self.sc: StandardScaler = model_info.sc
        self.pca: PCA = model_info.pca
        self.last_sample_id: int = model_info.last_sample_id
        self.redis_DB: DatabaseRedis = DatabaseRedis(model_id=self.ModelInfo.id)
        self.redis_DB.del_all_samples()
        self.required_column_names_list: List[str] = dp.read_required_column_names()
        # self.db: DatabaseSQLite = DatabaseSQLite() # todo usunąć

    def predict(self, sample_json: JSONType) -> Tuple[np.ndarray, np.ndarray]:
        sample_dict = json.loads(sample_json)

        transformed_sample_list_of_values = list(dp.transform_dict_row_in_one_hot_vectors_dict(sample_dict).values())

        transformed_sample_list_of_values = transformed_sample_list_of_values[3:]  # remove sale features from sample
        transformed_sample_list_of_values = self.pca.transform([transformed_sample_list_of_values])
        transformed_sample_list_of_values = self.sc.transform(transformed_sample_list_of_values)
        transformed_sample_list_of_values = normalize(transformed_sample_list_of_values, norm='l2')

        probability = self.model.predict_proba(transformed_sample_list_of_values).ravel()
        y = self.model.predict(transformed_sample_list_of_values)

        sample_dict['predicted'] = str(y[0])
        sample_dict['probabilities'] = json.dumps(list(probability))
        sample_json = json.dumps(sample_dict)

        self.redis_DB.rpush_sample(json_sample=sample_json)
        # self.db.insert_sample_as_dict(sample_dict) #todo usunąć, bo to evaluation server dodaje do sql

        return y, probability

if __name__ == '__main__':
    pass



