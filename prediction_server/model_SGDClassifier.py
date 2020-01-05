import numpy as np
import json
import pickle
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
        self.df_product_clicks_views = model_info.df_product_clicks_views
        # self.LabelEncoders_dict = None

        # if self.LabelEncoders_dict is None:
        #     file = open(
        #         f"/home/marcin/PycharmProjects/Engineering_Thesis/build_and_update_model_server/LabelEncoders_dict.pickle",
        #         "rb")
        #     self.LabelEncoders_dict = pickle.load(file)

        self.last_sample_id: int = model_info.last_sample_id
        self.redis_DB: DatabaseRedis = DatabaseRedis(model_id=self.ModelInfo.id)
        self.redis_DB.del_all_samples()
        # self.db: DatabaseSQLite = DatabaseSQLite() # todo usunąć

    def predict(self, sample_json: JSONType) -> Tuple[np.ndarray, np.ndarray]:
        sample_dict = json.loads(sample_json)
        sample_dict_result = sample_dict.copy()
        sample_dict.pop('sale', None)
        product_id = sample_dict['product_id']
        try:
            clicks, views = self.df_product_clicks_views.loc[self.df_product_clicks_views['product_id'] == product_id, ['clicks', 'views']].values.ravel()
            sample_dict['clicks'] = clicks
            sample_dict['views'] = views
            sample_dict['clicks_views_ratio'] = float(clicks/views)
        except ValueError:
            sample_dict['clicks'] = 0
            sample_dict['views'] = 0
            sample_dict['clicks_views_ratio'] = 0
        list_of_required_features = ['clicks_views_ratio', 'device_type', 'audience_id', 'partner_id',
                                     'product_brand', 'views', 'clicks', 'nb_clicks_1week', 'product_gender']
        sample_required_features = {key: sample_dict.get(key) for key in list_of_required_features}
        sample_list_of_features = list(sample_required_features.values())

        probabilities = self.model.predict_proba([sample_list_of_features])[0].ravel()
        if probabilities[0] > probabilities[1]:
            sample_dict_result['predicted'] = 0
        else:
            sample_dict_result['predicted'] = 1

        sample_dict_result['probabilities'] = json.dumps(list(probabilities))
        self.redis_DB.rpush_sample(json_sample=json.dumps(sample_dict_result))
        # self.db.insert_sample_as_dict(sample_dict) #todo usunąć, bo to evaluation server dodaje do sql

        return sample_dict_result['predicted'], probabilities

if __name__ == '__main__':
    pass



