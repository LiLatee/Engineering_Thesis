import numpy as np
import json
import pickle
import data_preprocessing as dp
# from client_SQLite import DatabaseSQLite
from model_info import ModelInfo
# from client_redis import DatabaseRedis

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
        self.standard_scaler = model_info.standard_scaler
        self.df_product_clicks_views = model_info.df_product_clicks_views

        self.last_sample_id: int = model_info.last_sample_id
        # self.redis_DB: DatabaseRedis = DatabaseRedis(model_id=self.ModelInfo.id)
        # self.redis_DB.del_all_samples()
        # self.db: DatabaseSQLite = DatabaseSQLite() # todo usunąć

    def predict(self, sample_json: JSONType) -> Tuple[np.ndarray, np.ndarray]:
        sample_dict = json.loads(sample_json)
        sample_dict_result = sample_dict.copy()
        sample_dict.pop('sale', None)
        product_id = sample_dict['product_id']


        # with open('clicks_views.txt', 'a+') as file:
        #     file.write((self.df_product_clicks_views.to_json(orient='records')))
        #     file.write('\n')
        try:
            clicks, views = self.df_product_clicks_views.loc[self.df_product_clicks_views['product_id'] == product_id, ['clicks', 'views']].values.ravel()
            # with open('pp.txt', 'a+') as file:
            #     file.write(str(clicks))
            #     file.write('\n')
            #     file.write(str(views))
            #     file.write('\n')

            sample_dict['clicks_views_ratio'] = float(clicks/views)
        except ValueError:
            sample_dict['clicks_views_ratio'] = 0
        list_of_required_features = ['product_price','clicks_views_ratio','device_type','audience_id','product_brand',
                                    'partner_id','product_gender','product_age_group','nb_clicks_1week']

        # with open('test.txt', 'a+') as file:
        #     file.write(str(sample_dict))
        #     file.write('\n')

        numerical_features_std = self.standard_scaler.transform([[sample_dict.get('nb_clicks_1week'), sample_dict.get('product_price')]]).ravel()
        sample_required_features = {key: sample_dict.get(key) for key in list_of_required_features}
        sample_required_features['nb_clicks_1week'] = numerical_features_std[0]
        sample_required_features['product_price'] = numerical_features_std[1]

        sample_list_of_features = list(sample_required_features.values())

        # with open('test2.txt', 'a+') as file:
        #     file.write(str(sample_list_of_features))
        #     file.write('\n')

        probabilities = self.model.predict_proba([sample_list_of_features])[0].ravel()
        if probabilities[0] > probabilities[1]:
            sample_dict_result['predicted'] = 0
        else:
            sample_dict_result['predicted'] = 1

        sample_dict_result['probabilities'] = json.dumps(list(probabilities))

        # self.redis_DB.rpush_sample(json_sample=json.dumps(sample_dict_result))
        # self.db.insert_sample_as_dict(sample_dict) #todo usunąć, bo to evaluation server dodaje do sql

        return sample_dict_result
        # return sample_dict_result['predicted'], probabilities

if __name__ == '__main__':
    pass



