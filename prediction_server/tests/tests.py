import unittest
import os
import pandas as pd
import numpy as np
import model_SGDClassifier


# start redis before testing
# sudo docker run --name redis_tests --network host --rm redis:latest --port 6379

class TestModelSGDClassifier(unittest.TestCase):

    def test_train_predict_update_predict(self):
        model = model_SGDClassifier.ModelSGDClassifier()

        self.create_and_save_model(model, 1000)
        # self.assertTrue(isinstance(model.pca, PCA))
        # self.assertTrue(isinstance(model.sc, StandardScaler))
        # self.assertTrue(isinstance(model.model, SGDClassifier))

        self.predict(model, 100)
        model.update_model()
        self.predict(model, 100)

        os.remove('data/sqlite3.db')

    def create_and_save_model(self, model, n_samples_for_training=1000):
        df = pd.read_csv(
            '/home/marcin/PycharmProjects/Engineering_Thesis/data_provider/data/CriteoSearchData-sorted-no-duplicates.csv',
            sep='\t',
            nrows=n_samples_for_training,
            skiprows=np.arange(1, 50001),
            header=0,
            low_memory=False)

        training_data_json = df.to_json(orient='records')
        model.create_model_and_save(training_data_json)

    def predict(self, model, n_samples_for_testing = 100):
        df = pd.read_csv(
            '/home/marcin/PycharmProjects/Engineering_Thesis/data_provider/data/CriteoSearchData-sorted-no-duplicates.csv',
            sep='\t',
            nrows=n_samples_for_testing,
            low_memory=False)

        y_pred = np.array([])
        y_test = np.array([])

        for id, row in df.iterrows():
            y, prob = model.predict(row.to_json())
            y_test = np.append(y_test, row['sale'])
            y_pred = np.append(y_pred, y)



if __name__ == '__main__':
    f = TestModelSGDClassifier()
    f.test_train_predict_update_predict()
