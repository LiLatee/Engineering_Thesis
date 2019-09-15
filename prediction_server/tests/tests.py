import unittest
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


import model_SGDClassifier


# start redis before testing
# sudo docker run --name redis_tests --network host --rm redis:latest --port 6379
# in client_SQLite.py change self.db_file in __init__() to "data/sqlite3.db"
# AND in client_redis change address to 127.0.0.1


class TestModelSGDClassifier(unittest.TestCase):

    def test_train_predict_update_predict(self, model):
        # model = model_SGDClassifier.ModelSGDClassifier()
        try:
            os.remove('data/sqlite3.db')
        except FileNotFoundError:
            pass

        self.create_and_save_model(model, 1000)
        self.assertTrue(isinstance(model.pca, PCA), "PCA is None")
        self.assertTrue(isinstance(model.sc, StandardScaler), "StandardScaler is None")
        self.assertTrue(isinstance(model.model, SGDClassifier), "SGDClassifier is None")

        self.predict(model, 100)
        model.update_model()
        self.predict(model, 100)


    def create_and_save_model(self, model, N_SAMPLES_FOR_TRAINING=1000):
        df = pd.read_csv(
            '/home/marcin/PycharmProjects/Engineering_Thesis/data_provider/data/CriteoSearchData-sorted-no-duplicates.csv',
            sep='\t',
            nrows=N_SAMPLES_FOR_TRAINING,
            skiprows=np.arange(1, 50001),
            header=0,
            low_memory=False)

        training_data_json = df.to_json(orient='records')
        model.create_model_and_save(training_data_json)

    def predict(self, model, N_SAMPLES_FOR_TESTING = 100):
        df = pd.read_csv(
            '/home/marcin/PycharmProjects/Engineering_Thesis/data_provider/data/CriteoSearchData-sorted-no-duplicates.csv',
            sep='\t',
            nrows=N_SAMPLES_FOR_TESTING,
            low_memory=False)

        y_pred = np.array([])
        y_test = np.array([])

        for id, row in df.iterrows():
            y, prob = model.predict(row.to_json())
            y_test = np.append(y_test, row['sale'])
            y_pred = np.append(y_pred, y)



if __name__ == '__main__':
    model = model_SGDClassifier.ModelSGDClassifier()
    f = TestModelSGDClassifier()
    f.test_train_predict_update_predict(model)
    f.predict(model, 100)
    model.update_model()

