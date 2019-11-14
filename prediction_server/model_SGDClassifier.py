import pandas as pd
import numpy as np
import json
import collections
import time
import requests
import pickle

import data_preprocessing as dp
from client_SQLite import DatabaseSQLite
from model_info import ModelInfo
from client_redis import DatabaseRedis

from typing import List, Dict, Union, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, normalize
from imblearn.over_sampling import ADASYN
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score

# JSONType = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONType = Union[str, bytes, bytearray]
RowAsDictType = Dict[str, Union[str, float, int]]


class ModelSGDClassifier:

    def __init__(self, model_info: ModelInfo = None) -> None:
        if model_info is None:
            self.model: SGDClassifier = None
            self.sc: StandardScaler = None
            self.pca: PCA = None
            self.last_sample_id: int = -1
            self.load_model_if_exists()
        else:
            self.ModelInfo: ModelInfo = model_info
            self.model: SGDClassifier = model_info.model
            self.sc: StandardScaler = model_info.sc
            self.pca: PCA = model_info.pca
            self.last_sample_id: int = model_info.last_sample_id

        self.required_column_names_list: List[str] = dp.read_required_column_names()
        self.redis_DB: DatabaseRedis = DatabaseRedis(model_id=self.ModelInfo.id)
        self.db: DatabaseSQLite = DatabaseSQLite()
        self.redis_DB.del_all_samples()

    def create_model_and_save(self, training_data_json: JSONType) -> None:
        print("create_model_and_save")

        start = time.time()
        self.pca, self.sc, x_test_std, x_train_std, y_test, y_train = dp.create_train_and_test_sets(training_data_json)
        end = time.time()
        print('Czas tworzenia zbiorów: {0}'.format((end-start)))
        # ppn = Perceptron(eta0=0.1, random_state=1, n_jobs=-1)
        # ppn.fit(X_train_std, y_train)
        # lr = LogisticRegression(C=1000.0, random_state=1, solver="lbfgs", n_jobs=-1, verbose=1)
        # lr.fit(X_train_std, y_train)

        # X_train_std, X_val_std, y_train, y_val = train_test_split(X_train_std, y_train, test_size=0.2, random_state=1)
        lr = SGDClassifier(loss='log', random_state=1, tol=1e-3, max_iter=1000, penalty='l1', alpha=1e-05, n_jobs=-1)
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
        self.save_model() # todo tylko jak sqlite

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
        self.db.insert_sample_as_dict(sample_dict) #todo usunąć, bo to evaluation server dodaje do sql

        return y, probability

    def update_model(self) -> None:
        # samples_list_of_dicts = self.db.get_samples_to_update_model_as_list_of_dicts(self.last_sample_id)
        response = requests.request(method="GET", url='http://sqlite_api:8764/samples/?last_model_id=' + str(self.last_sample_id))
        samples_list_of_dicts = json.loads(response.content) #todo

        x, y = dp.split_data_to_x_and_y(samples_list_of_dicts)
        x = self.pca.transform(x)
        adasyn = ADASYN(random_state=1)
        x, y = adasyn.fit_resample(x, y) #TODO blad jak wszystkie probki sa z jednej klasy
        x = self.sc.transform(x)
        x = normalize(x, norm='l2')
        y = np.array([int(i) for i in y])

        self.model.partial_fit(x, y, classes=np.array([0, 1]))
        self.save_model() # todo zostawić tylko przy używaniu sqlite

        self.last_sample_id = samples_list_of_dicts[-1]['id'] #todo sprawdzić czy działa

        print("LOG: updating model DONE")

    # def transform_list_of_jsons_to_list_of_one_hot_vectors_dicts(self, samples_list_of_jsons) -> List:
    #     samples = []
    #     for sample_as_json in samples_list_of_jsons:
    #         samples.append(self.transform_dict_row_in_one_hot_vectors_dict(json.loads(sample_as_json)))
    #
    #     return samples

    def save_model(self) -> None:
        # if self.model is None:
        #     print("LOG: " + "There is not model available. Must be created.")

        # new_version = self.db.get_last_version_of_specified_model('SGDClassifier') + 1 # todo usunąć hardcoded nazwę modelu
        response = requests.request(method="GET", url='http://sqlite_api:8764/models/?model_name=SGDClassifier')
        new_version = pickle.loads(response.content) #todo

        # zapisywanie modelu do sqlite
        model_info = ModelInfo()
        model_info.name = 'SGDClassifier'
        model_info.version = new_version
        model_info.date_of_create = time.time()
        model_info.last_sample_id = self.last_sample_id
        model_info.model = self.model
        model_info.sc = self.sc
        model_info.pca = self.pca
        # self.db.insert_ModelInfo(model_info) #todo
        requests.request(method='POST', url='http://sqlite_api:8764/models', data=pickle.dumps(model_info))

        print("LOG: saving model DONE")

        # zapisywanie modelu do pliku
        # current_dir = os.path.dirname(__file__)
        # dest = os.path.join('pickle_objects')
        # if not os.path.existsaki po prostu jest(dest):
        #     os.makedirs(dest)
        # pickle.dump(self.model, open(os.path.join(dest, "SGDClassifier.pkl"), mode='wb'), protocol=4)
        # print("LOG: " + "model saved in directory: " + current_dir + '\\' + dest + '\SGDClassifier.pkl')


        # zapisywanie do cassandry
        # db = CassandraClient()
        # last_sample_id = db.get_last_sample_id()

        # None, "SGDClassifier", 0, None, last_sample_id, self.model, self.sc, self.pca
        # models = ModelHistory(
        #     id=-1,
        #     name='SGDClassifier',
        #     version=0,
        #     creation_timestamp='-1',
        #     model=pickle.dumps(self.model),
        #     standard_scaler=pickle.dumps(self.sc),
        #     pca=pickle.dumps(self.pca))

        # pca_binary = pickle.dumps(self.pca)
        # pca_bytes = len(pca_binary)
        # print('TUUUUUUUU')
        # print(pca_bytes)
        # models = {
        #     "id": -1,
        #     "name": "SGDClassifier",
        #     "version": 0,
        #     "timestamp":  1598891820,
        #     # "timestamp":  '2016-04-06 13:06:11.534',
        #     "model": pickle.dumps(self.model),
        #     "standard_scaler": pickle.dumps(self.sc),
        #     "pca_one": pca_binary[:1000000],
        #     "pca_two": pca_binary[1000000:2000000]
        # }
        #
        # db.insert_models(models)
        # print("LOG: saving model DONE")

    def load_model_if_exists(self) -> None:
        # wczytywanie z sqlite
        # model_info = self.db.get_last_ModelInfo() #todo
        response = requests.request(method='GET', url='http://sqlite_api:8764/models/get_last')
        model_info = pickle.loads(response.content)

        if model_info is None:
            return
        self.model = model_info.model
        self.sc = model_info.sc
        self.pca = model_info.pca

        print("LOG: " + "Model loaded.")

        # wczytywanie z pliku
        # current_dir = os.path.dirname(__file__)
        # self.model = pickle.load(open(os.path.join(current_dir, 'pickle_objects', 'SGDClassifier.pkl'), mode='rb'))
        # print("LOG: " + "model load from directory: " + current_dir + '\SGDClassifier.pkl')

        # wczytywanie z cassandry
        # db = CassandraClient()
        # models = db.get_last_models()
        # self.model = pickle.loads(models.model)
        # self.sc = pickle.loads(models.standard_scaler)
        # self.pca = pickle.loads(models.pca_one+models.pca)

if __name__ == '__main__':
    pass



