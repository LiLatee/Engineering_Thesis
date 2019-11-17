import numpy as np
import json
import collections
import time
import requests
import pickle

import data_preprocessing as dp
from model_info import ModelInfo
from client_cass import CassandraClient
from typing import List, Dict, Union, Any, Tuple
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, normalize
from imblearn.over_sampling import ADASYN
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score

# JSONType = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONType = Union[str, bytes, bytearray]
RowAsDictType = Dict[str, Union[str, float, int]]


class ModelSGDClassifier:

    def __init__(self) -> None:
        self.model: SGDClassifier = None
        self.sc: StandardScaler = None
        self.pca: PCA = None
        self.last_sample_id: int = -1
        self.load_last_model()

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

    def update_model(self) -> None:
        # response = requests.request(method="GET", url='http://sqlite_api:8764/samples-for-update/?last_sample_id=' + str(self.last_sample_id))
        cass = CassandraClient()
        samples_list_of_dicts = cass.get_samples_for_update_model_as_list_of_dicts(self.last_sample_id)
        # response = requests.request(method="GET", url='http://cassandra_api:9042/samples-for-update/?last_sample_id=' + str(self.last_sample_id))
        # samples_list_of_dicts = json.loads(response.content) #todo

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

    def save_model(self) -> None:
        response = requests.request(method="GET", url='http://sqlite_api:8764/models/?model_name=SGDClassifier')  # todo usunąć hardcoded nazwę modelu
        new_version = pickle.loads(response.content)

        # zapisywanie modelu do sqlite
        model_info = ModelInfo()
        model_info.name = 'SGDClassifier'
        model_info.version = new_version
        model_info.date_of_create = time.time()
        model_info.last_sample_id = self.last_sample_id
        model_info.model = self.model
        model_info.sc = self.sc
        model_info.pca = self.pca
        requests.request(method='POST', url='http://sqlite_api:8764/models', data=pickle.dumps(model_info))

        print("LOG: saving model DONE")

    def load_last_model(self) -> None:
        # wczytywanie z sqlite
        response = requests.request(method='GET', url='http://sqlite_api:8764/models/get_last')
        model_info = pickle.loads(response.content)

        if model_info is None:
            return
        self.model = model_info.model
        self.sc = model_info.sc
        self.pca = model_info.pca

        print("LOG: " + "Model loaded.")


if __name__ == '__main__':
    pass



