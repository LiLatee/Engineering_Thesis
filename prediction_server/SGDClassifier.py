import pandas as pd
import numpy as np
import pickle
import os
import json
import collections

import cass_client as db
# import DatabaseSQLite
import ModelInfo
import time

from DatabaseRedis import DatabaseRedis
from typing import List, Dict, NoReturn, Union, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, normalize
from imblearn.over_sampling import ADASYN
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score

import redis
from rq import Queue
import requests

JSONType = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
# JSONType = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONType = Union[str, bytes, bytearray]
RowAsDictType = Dict[str, Union[str, float, int]]


class ModelSGDClassifier:

    def __init__(self) -> None:
        self.model: SGDClassifier = None
        self.sc: StandardScaler = None
        # self.df_original: pd.DataFrame = None
        self.pca = None
        self.last_sample_id: Optional[int] = None
        self.required_column_names_list: List[str] = self.read_required_column_names()
        self.redis_DB = DatabaseRedis()

    def create_model_and_save(self, json_training_data: JSONType) -> None:
        # print("create_model_and_save")

        df_original = pd.read_json(json_training_data) # TODO wywalic DATAFRAME
        # df_one_hot_vectors = df_one_hot_vectors.dropna(axis=0)  # usuwanie wierszy, które zawierają null

        start = time.time()
        x_test_std, x_train_std, y_test, y_train = self.create_train_and_test_sets(df_original)
        end = time.time()
        print('Czas tworzenia zbiorów: {0}'.format((end-start)))
        # ppn = Perceptron(eta0=0.1, random_state=1, n_jobs=-1)
        # ppn.fit(X_train_std, y_train)

        # lr = LogisticRegression(C=1000.0, random_state=1, solver="lbfgs", n_jobs=-1, verbose=1)
        # lr.fit(X_train_std, y_train)

        # X_train_std, X_val_std, y_train, y_val = train_test_split(X_train_std, y_train, test_size=0.2, random_state=1)
        lr = SGDClassifier(loss='log', random_state=1, tol=1e-3, max_iter=1000, penalty='l1', alpha=1e-05, n_jobs=-1)
        # lr = SGDClassifier(loss='log', verbose=1, n_jobs=-1, random_state=1, tol=1e-3, max_iter=1000, penalty='l2', alpha=1e-20)
        start = time.time()
        lr.fit(x_train_std, y_train)
        end = time.time()
        print('Czas treningu: {0}'.format((end-start)))
        print("LOG: creating model DONE")

        print(collections.Counter(y_test))
        y_pred = lr.predict(x_test_std)
        print(collections.Counter(y_pred))

        print("LOG: testing model DONE")
        print('balanced_accuracy_score: {0}'.format(balanced_accuracy_score(y_test, y_pred)))
        print('accuracy_score: {0}'.format(accuracy_score(y_test, y_pred)))

        print('Nieprawidłowo sklasyfikowane próbki: %d' % (y_test != y_pred).sum())

        print('classification_report :\n', classification_report(y_test, y_pred))
        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        print(confmat)

        self.model = lr
        self.save_model()

    def create_train_and_test_sets(self, df: pd.DataFrame) -> List[np.ndarray]:
        # print("create_train_and_test_sets")
        list_of_dicts_of_samples = self.transform_df_into_list_of_one_hot_vectors_dicts(df)

        # x = []
        # y = []
        # for s in array_of_dicts_of_samples:
        #     x.append(list(s.values())[3:])
        #     y.append(list(s.values())[:1][0])

        x = [list(s.values())[3:] for s in list_of_dicts_of_samples]
        y = [list(s.values())[:1][0] for s in list_of_dicts_of_samples]
        x = np.array(x)
        y = np.array(y)

        # x = df_one_hot_vectors.iloc[:, 3:].values
        # y = df_one_hot_vectors.iloc[:, :1].values.ravel()  # tutaj powinny być chyba 3 kolumny

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

        print('Liczba etykiet w zbiorze y:', np.bincount(y))
        print('Liczba etykiet w zbiorze y_train:', np.bincount(y_train))
        print('Liczba etykiet w zbiorze y_test:', np.bincount(y_test))

        self.pca = PCA(n_components=500, random_state=1)
        x_train = self.pca.fit_transform(x_train)
        x_test = self.pca.transform(x_test)

        adasyn = ADASYN(random_state=1)
        x_train, y_train = adasyn.fit_resample(x_train, y_train)

        self.sc = StandardScaler()
        self.sc.fit(x_train)
        x_train_std = self.sc.transform(x_train)
        x_train_std = normalize(x_train_std, norm='l2')
        x_test_std = self.sc.transform(x_test)
        x_test_std = normalize(x_test_std, norm='l2')

        return [x_test_std, x_train_std, y_test, y_train]

    def transform_df_into_list_of_one_hot_vectors_dicts(self, df_to_transform: pd.DataFrame) -> List:
        # print("transform_df_into_df_with_one_hot_vectors")
        data_as_dict = json.loads(df_to_transform.T.to_json())

        samples = []
        for row_number, row_as_dict in data_as_dict.items():
            json_row = json.dumps(row_as_dict)
            samples.append(self.transform_json_row_in_one_hot_vectors_dict(json_row))



        # df = pd.DataFrame(samples)
        # df = df.fillna(0)

        return samples

    def transform_json_row_in_one_hot_vectors_dict(self, row_as_json: JSONType) -> Dict[str, Union[str, float, int]]:
        # print('transform_json_row_in_one_hot_vectors_dict')
        row_as_dict = json.loads(row_as_json)

        transformed_row_as_dict = self.create_dict_as_transformed_row_and_set_no_one_hot_vectors_columns(row_as_dict)
        transformed_row_as_dict = self.set_values_to_one_hot_vectors_columns(row_as_dict, transformed_row_as_dict)

        return transformed_row_as_dict

    def create_dict_as_transformed_row_and_set_no_one_hot_vectors_columns(self, old_dict: RowAsDictType) -> RowAsDictType:
        # print('create_dict_as_transformed_row_and_set_no_one_hot_vectors_columns')

        new_dict = dict.fromkeys(self.required_column_names_list, 0)
        new_dict['Sale'] = old_dict['Sale']
        new_dict['SalesAmountInEuro'] = old_dict['SalesAmountInEuro']
        new_dict['time_delay_for_conversion'] = old_dict[
            'time_delay_for_conversion']
        new_dict['click_timestamp'] = old_dict['click_timestamp']
        new_dict['nb_clicks_1week'] = old_dict['nb_clicks_1week']

        return new_dict

    def set_values_to_one_hot_vectors_columns(self, old_dict: RowAsDictType, new_dict: RowAsDictType) -> RowAsDictType:
        # print('set_values_to_one_hot_vectors_columns')

        for column_number, (column_name, cell_value) in enumerate(old_dict.items()):
            if column_number > 2:
                transformed_column_name = column_name + '_' + str(cell_value)
                if transformed_column_name in self.required_column_names_list:
                    new_dict[transformed_column_name] = 1

        return new_dict

    @staticmethod
    def read_required_column_names() -> List[str]:
        required_column_name_file = open('required_column_names_list.txt', 'r')
        required_column_names_list = required_column_name_file.read().splitlines()
        return required_column_names_list

    @staticmethod
    def replace_none_values_in_dict(dict_to_change: RowAsDictType, value_for_none):
        result = {}
        for k, v in dict_to_change.items():
            if v is None:
                result[k] = value_for_none #TODO zmienić aby wszystko w dataframeach i slownikach bylo str
            else:
                result[k] = v
        return result

    def predict(self, sample_json: JSONType) -> Tuple[np.ndarray, np.ndarray]:

        transformed_sample_dict = list(self.transform_json_row_in_one_hot_vectors_dict(sample_json).values())

        transformed_sample_dict = transformed_sample_dict[3:]  # remove sales features from sample
        transformed_sample_dict = self.pca.transform([transformed_sample_dict])
        transformed_sample_dict = self.sc.transform(transformed_sample_dict)
        transformed_sample_dict = normalize(transformed_sample_dict, norm='l2')

        probability = self.model.predict_proba(transformed_sample_dict).ravel()
        y = self.model.predict(transformed_sample_dict)


        sample_dict = json.loads(sample_json)
        sample_dict['predicted'] = str(y[0])
        sample_dict['probabilities'] = list(probability)
        sample_json = json.dumps(sample_dict)


        self.redis_DB.rpush_sample(sample_json)


        # db = DatabaseSQLite.DatabaseSQLite()
        # db.add_row_from_json(sample_json=sample_json)

        return y, probability

    def update_model(self) -> None:
        pass
        # # db = DatabaseSQLite.DatabaseSQLite()
        # # df_samples_to_update = db.get_samples_to_update_model_as_df()
        # # list_of_dicts_of_samples = self.transform_df_into_list_of_one_hot_vectors_dicts(df_samples_to_update)
        #
        # samples_list_of_bytes = self.redis_DB.get_all_samples_as_list_of_bytes()
        # list_of_dicts_of_samples = self.transform_list_of_dicts_to_list_of_one_hot_vectors_dicts(samples_list_of_bytes)
        #
        # # df_one_hot_vectors = df_one_hot_vectors.dropna(axis=0)  # usuwanie wierszy, które zawierają null
        #
        # # x = []
        # # y = []
        # # for s in array_of_dicts_of_samples:
        # #     x.append(list(s.values())[3:])
        # #     y.append(list(s.values())[:1][0])
        #
        # x = [list(s.values())[3:] for s in list_of_dicts_of_samples]
        # y = [list(s.values())[:1][0] for s in list_of_dicts_of_samples]
        # x = np.array(x)
        # y = np.array(y)
        # # x = df_one_hot_vectors.iloc[:, 3:].values
        # # y = df_one_hot_vectors['Sale'].values.ravel()
        # print(x)
        # x = self.pca.transform(x)
        # adasyn = ADASYN(random_state=1)
        # x, y = adasyn.fit_resample(x, y) #TODO blad jak wszystkie probki sa z jedne klasy
        # x = self.sc.transform(x)
        # x = normalize(x, norm='l2')
        # y = np.array([int(i) for i in y])
        #
        # self.model.partial_fit(x, y, classes=np.array([0, 1]))
        # self.save_model()
        # print("LOG: updating model DONE")

    def transform_list_of_dicts_to_list_of_one_hot_vectors_dicts(self, samples_list_of_jsons) -> List:
        # print("transform_df_into_df_with_one_hot_vectors")
        # data_as_dict = json.loads(df_to_transform.T.to_json())

        samples = []
        for sample_as_json in samples_list_of_jsons:
            samples.append(self.transform_json_row_in_one_hot_vectors_dict(sample_as_json))



        # df = pd.DataFrame(samples)
        # df = df.fillna(0)

        return samples

    def save_model(self) -> None:
        pass
        # if self.model is None:
        #     print("LOG: " + "There is not model available. Must be created.")
        # # zapisywanie modelu do pliku
        # # current_dir = os.path.dirname(__file__)
        # # dest = os.path.join('pickle_objects')
        # # if not os.path.exists(dest):
        # #     os.makedirs(dest)
        # # pickle.dump(self.model, open(os.path.join(dest, "SGDClassifier.pkl"), mode='wb'), protocol=4)
        # # print("LOG: " + "model saved in directory: " + current_dir + '\\' + dest + '\SGDClassifier.pkl')
        #
        # db = DatabaseSQLite.DatabaseSQLite()
        # last_sample_id = db.get_last_sample_id()
        # model_info = ModelInfo.ModelInfo(None, "SGDClassifier", 0, None, last_sample_id, self.model, self.sc, self.pca)
        # db.add_model(model_info)
        # print("LOG: saving model DONE")

    def load_model(self) -> None:
        pass
        # # current_dir = os.path.dirname(__file__)
        # # self.model = pickle.load(open(os.path.join(current_dir, 'pickle_objects', 'SGDClassifier.pkl'), mode='rb'))
        # # print("LOG: " + "model load from directory: " + current_dir + '\SGDClassifier.pkl')
        # db = DatabaseSQLite.DatabaseSQLite()
        # model_info = db.get_last_model_info()
        # self.model = model_info.model
        # self.sc = model_info.sc
        # self.pca = model_info.pca
        #
        # print("LOG: " + "Model loaded.")

    def test_predict_1k(self):
        print("Testing...")
        df = pd.read_csv('/home/marcin/PycharmProjects/Engineering_Thesis/dataset/CriteoSearchData-sorted-no-duplicates.csv',
                         sep='\t',
                         nrows=1000,
                         low_memory=False)


        y_pred = np.array([])
        y_test = np.array([])

        for id, row in df.iterrows():
            y, prob = self.predict(row.to_json())
            y_test = np.append(y_test, row['Sale'])
            y_pred = np.append(y_pred, y)


        # end = time.time()
        # print('testing: {0}'.format((end-start)))
        # print(collections.Counter(y_test))
        # print(collections.Counter(y_pred))
        #
        print("LOG: testing model DONE")
        print('balanced_accuracy_score: {0}'.format(balanced_accuracy_score(y_test, y_pred)))
        print('accuracy_score: {0}'.format(accuracy_score(y_test, y_pred)))

        print('Nieprawidłowo sklasyfikowane próbki: %d' % (y_test != y_pred).sum())

        print('classification_report :\n', classification_report(y_test, y_pred))
        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        print(confmat)

    def test_train(self, n_samples_for_training: int) -> None:
        df = pd.read_csv('/home/marcin/PycharmProjects/Engineering_Thesis/dataset/CriteoSearchData-sorted-no-duplicates.csv',
                         sep='\t',
                         nrows=n_samples_for_training,
                         skiprows=np.arange(1,50001),
                         header=0,
                         low_memory=False)

        training_data_json = df.to_json()

        print("Training...")
        start = time.time()
        self.create_model_and_save(training_data_json)
        end = time.time()
        print('CALOSC TRENIG: {0}'.format((end-start)))
        print("DONE")


        # tests = []
        # for id, row in df[1001:2000].iterrows():
        #     transformed_x = self.transform_json_row_in_one_hot_vectors_list_of_values(row.to_json())
        #     transformed_x = transformed_x[3:]  # remove sales features from sample
        #     tests.append(transformed_x)
        # tests = np.asarray(tests)
        # tests = self.sc.transform(tests)
        # y = self.model.predict(tests)
        # probability = self.model.predict_proba(tests)
        # print("y: %s\t probability: %s" % (str(y), str(probability)))



if __name__ == '__main__':
    m = ModelSGDClassifier()
    # m.test_train(n_samples_for_training=10000)
    m.load_model()
    m.test_predict_1k()
    # m.update_model()

