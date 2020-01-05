import numpy as np
import json
import collections
import time
import requests
import pickle
import pandas as pd

import data_preprocessing as dp
from model_info import ModelInfo
from client_cass import CassandraClient
from typing import List, Dict, Union, Any, Tuple
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, normalize
from imblearn.over_sampling import ADASYN
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

# JSONType = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONType = Union[str, bytes, bytearray]
RowAsDictType = Dict[str, Union[str, float, int]]


class ModelSGDClassifier:

    def __init__(self) -> None:
        # self.names_of_columns_with_ids = ['audience_id', 'device_type',
        #                                   'partner_id', 'product_age_group', 'product_brand',
        #                                   'product_category_1', 'product_category_2', 'product_category_3',
        #                                   'product_category_4', 'product_category_5', 'product_category_6',
        #                                   'product_category_7', 'product_country', 'product_gender', 'product_id',
        #                                   'product_title', 'user_id']
        self.model: SGDClassifier = None
        self.last_sample_id: str = None
        self.df_product_clicks_views = None
        # self.LabelEncoders_dict = None
        self.load_last_model()

        # if self.LabelEncoders_dict is None:
        #     file = open(
        #         f"/home/marcin/PycharmProjects/Engineering_Thesis/build_and_update_model_server/LabelEncoders_dict.pickle",
        #         "rb")
        #     self.LabelEncoders_dict = pickle.load(file)

    def create_model_and_save(self, training_data_json: JSONType, update: bool = False) -> None:
        print("create_model_and_save")
        start = time.time()

        df = pd.read_json(training_data_json, orient='records')
        if not update:
            cass = CassandraClient(table_name='all_stored_samples')
            for _, row in df.iterrows():
                cass.insert_sample(row.to_dict())
        else:
            df = df[['sale','sales_amount_in_euro','time_delay_for_conversion','click_timestamp','nb_clicks_1week','product_price',
                    'product_age_group','device_type','audience_id','product_gender','product_brand','product_category_1',
                    'product_category_2','product_category_3','product_category_4','product_category_5','product_category_6',
                    'product_category_7','product_country','product_id','product_title','partner_id','user_id']]
        # df[self.names_of_columns_with_ids] = df[self.names_of_columns_with_ids].astype(str)
        # df[self.names_of_columns_with_ids] = df[self.names_of_columns_with_ids].apply(
        #     lambda x: self.LabelEncoders_dict[x.name].transform(x))

        df['day_of_week'] = pd.to_datetime(df['click_timestamp'], unit='s').dt.weekday
        df['hour'] = pd.to_datetime(df['click_timestamp'], unit='s').dt.hour

        df_product_clicks = df[['sale', 'product_id']]
        df_product_clicks = df_product_clicks.groupby('product_id').sum().reset_index()
        df = pd.merge(df,
                      df_product_clicks,
                      left_on='product_id',
                      right_on='product_id',
                      how='left')
        df = df.rename(columns={'sale_x': 'sale', 'sale_y': 'clicks'})

        df_product_views = df['product_id'].value_counts().to_frame("views").reset_index()
        df_product_views = df_product_views.rename(columns={'index': 'product_id'})
        df = pd.merge(df,
                      df_product_views,
                      left_on='product_id',
                      right_on='product_id',
                      how='left')
        df = df.rename(columns={'sale_x': 'sale', 'sale_y': 'views'})

        df['clicks_views_ratio'] = df['clicks'] / df['views']
        df.loc[(df['views'] < 5), 'clicks_views_ratio'] = 0

        df_product_clicks_views = df[['product_id', 'clicks', 'views']]
        self.df_product_clicks_views = df_product_clicks_views

        number_of_samples = df.shape[0]
        counter = collections.Counter(df['sale'])
        percent_of_ones = counter[1] / number_of_samples

        model = SGDClassifier(loss='log', random_state=1, tol=1e-3, max_iter=1000, penalty='l1', alpha=1e-05, n_jobs=-1,
                              class_weight={0: percent_of_ones, 1: 1 - percent_of_ones})
        x_train = df[
            ['clicks_views_ratio', 'device_type', 'audience_id', 'partner_id', 'product_brand', 'views', 'clicks',
             'nb_clicks_1week', 'product_gender']]
        y_train = df['sale']

        # x_train = df.loc[:, df.columns != 'sale'] # todo wywalić timestamp df.loc[:,~df.columns.isin(['sale','click_timestamp])]
        # y_train = df['sale']

        model.fit(x_train, y_train)


        end = time.time()
        print('Czas treningu: {0}'.format((end - start)))
        print("LOG: creating model DONE")


        # print(collections.Counter(y_test))
        # y_pred = lr.predict(x_test_std)
        # print(collections.Counter(y_pred))
        #
        # print("LOG: testing model DONE")
        # print('balanced_accuracy_score: {0}'.format(balanced_accuracy_score(y_test, y_pred)))
        # print('accuracy_score: {0}'.format(accuracy_score(y_test, y_pred)))
        #
        # print('Nieprawidłowo sklasyfikowane próbki: %d' % (y_test != y_pred).sum())
        #
        # print('classification_report :\n', classification_report(y_test, y_pred))
        # confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        # print(confmat)

        self.model = model
        self.save_model()  # todo tylko jak sqlite

    def update_model(self) -> None:
        from uuid import UUID

        class UUIDEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, UUID):
                    # if the obj is uuid, we simply return the value of uuid
                    return obj.hex
                return json.JSONEncoder.default(self, obj)

        cass = CassandraClient(table_name='all_stored_samples')
        samples_list_of_dicts = cass.get_all_samples_as_list_of_dicts() # todo tu są tylko brane dodane i omijane te ze zbioru treningowego, w trakcie
        self.create_model_and_save(json.dumps(samples_list_of_dicts, cls=UUIDEncoder), update=True)
        print("LOG: updating model DONE")

    def save_model(self) -> None:
        response = requests.request(method="GET",
                                    url='http://sqlite_api:8764/models/?model_name=SGDClassifier')  # todo usunąć hardcoded nazwę modelu
        new_version = int(pickle.loads(response.content)) + 1

        # zapisywanie modelu do sqlite
        model_info = ModelInfo()
        model_info.name = 'SGDClassifier'
        model_info.version = new_version
        model_info.date_of_create = time.time()
        model_info.last_sample_id = self.last_sample_id
        model_info.model = self.model
        model_info.df_product_clicks_views = self.df_product_clicks_views
        # model_info.LabelEncoders_dict = self.LabelEncoders_dict

        requests.request(method='POST', url='http://sqlite_api:8764/models', data=pickle.dumps(model_info))

        print("LOG: saving model DONE")

    def load_last_model(self) -> None:
        # wczytywanie z sqlite
        response = requests.request(method='GET', url='http://sqlite_api:8764/models/get_last')
        model_info = pickle.loads(response.content)

        if model_info is None:
            return
        self.model = model_info.model
        self.df_product_clicks_views = model_info.df_product_clicks_views # todo niepotrzebne?
        # self.LabelEncoders_dict = model_info.LabelEncoders_dict

        print("LOG: " + "Model loaded.")

    # def create_LabelEncoders_dict_for_each_column(self, df):
    #     LabelEncoders_dict = collections.defaultdict(LabelEncoder)
    #
    #     df[self.names_of_columns_with_ids] = df[self.names_of_columns_with_ids].astype(str)
    #     # Encoding the variable
    #     df[self.names_of_columns_with_ids] = df[self.names_of_columns_with_ids].apply(
    #         lambda x: LabelEncoders_dict[x.name].fit_transform(x))
    #
    #     return LabelEncoders_dict


if __name__ == '__main__':
    pass
