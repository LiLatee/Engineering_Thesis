import numpy as np
import json
import collections
import time
import requests
import pickle
import pandas as pd
from sklearn.model_selection import cross_validate

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
        self.model: SGDClassifier = None
        self.last_sample_id: str = None
        self.df_product_clicks_views = None
        self.sc = None
        # self.load_last_model()

    def show_mean_scores(self, scores):
        print("test_acc: %0.2f (+/- %0.2f)" % (scores['test_acc'].mean(), scores['test_acc'].std()))
        print("train_acc: %0.2f (+/- %0.2f)" % (scores['train_acc'].mean(), scores['train_acc'].std()))
        print("test_bal_acc: %0.2f (+/- %0.2f)" % (scores['test_bal_acc'].mean(), scores['test_bal_acc'].std()))
        print("train_bal_acc: %0.2f (+/- %0.2f)" % (scores['train_bal_acc'].mean(), scores['train_bal_acc'].std()))
        print("test_f1: %0.2f (+/- %0.2f)" % (scores['test_f1'].mean(), scores['test_f1'].std()))
        print("train_f1: %0.2f (+/- %0.2f)" % (scores['train_f1'].mean(), scores['train_f1'].std()))
        print("test_recall: %0.2f (+/- %0.2f)" % (scores['test_recall'].mean(), scores['test_recall'].std()))
        print("train_recall: %0.2f (+/- %0.2f)" % (scores['train_recall'].mean(), scores['train_recall'].std()))
        print("test_average_precision: %0.2f (+/- %0.2f)" % (
        scores['test_average_precision'].mean(), scores['test_average_precision'].std()))
        print("train_average_precision: %0.2f (+/- %0.2f)" % (
        scores['train_average_precision'].mean(), scores['train_average_precision'].std()))
        print("test_roc_auc: %0.2f (+/- %0.2f)" % (scores['test_roc_auc'].mean(), scores['test_roc_auc'].std()))
        print("train_roc_auc: %0.2f (+/- %0.2f)" % (scores['train_roc_auc'].mean(), scores['train_roc_auc'].std()))


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

        df_product_clicks_views = df[['product_id', 'clicks', 'views']].drop_duplicates()
        self.df_product_clicks_views = df_product_clicks_views


        # with open('test.txt', 'a+') as f:
        #     f.write(str(df.shape[0]))
        #     f.write('\n')
        #     f.write(df.to_json(orient='records'))
        #     f.write('\n')
        #     f.write('FFFFF')
        #     f.write('\n')


        self.sc = StandardScaler()
        df[['nb_clicks_1week', 'product_price']] = self.sc.fit_transform(df[['nb_clicks_1week', 'product_price']].to_numpy())
        df = df.sort_values(by=['click_timestamp'], ascending=True)

        # import uuid
        # df.to_csv(str(df.shape[0]) + '_model_' + str(uuid.uuid1()) + '.csv', sep=',', index=False)
        # with open(str(uuid.uuid1())+'.txt', 'a+') as f:
        #     f.write(str(df.shape[0]))
        #     f.write('\n')
        #     f.write(df.to_json(orient='records'))
        #     f.write('\n')
        #     f.write('FFFFF')
        #     f.write('\n')


        x_train = df[['product_price', 'clicks_views_ratio', 'device_type', 'audience_id', 'product_brand',
                      'partner_id', 'product_gender', 'product_age_group', 'nb_clicks_1week']]
        y_train = df['sale']

        number_of_samples = df.shape[0]
        counter = collections.Counter(df['sale'])
        percent_of_ones = counter[1] / number_of_samples
        model = SGDClassifier(loss='log', max_iter=100, penalty='l1', alpha=0.0001, n_jobs=-1,
                              class_weight={0: percent_of_ones, 1: 1 - percent_of_ones})

        scoring = {
            'acc': 'accuracy',
            'bal_acc': 'balanced_accuracy',
            'roc_auc': 'roc_auc',
            'f1': 'f1',
            'recall': 'recall',
            'precision': 'precision',
            'average_precision': 'average_precision'
        }
        scores = cross_validate(model, x_train, y_train, cv=5, scoring=scoring, return_train_score=True)
        print("WYNIKI TRENINGU:\n")
        self.show_mean_scores(scores)

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
        samples_list_of_dicts = cass.get_all_samples_as_list_of_dicts()
        # samples_list_of_dicts = cass.get_last_n_samples_as_list_of_dicts(100000)
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
        model_info.standard_scaler = self.sc
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
        self.sc = model_info.standard_scaler
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
