import pandas as pd
import numpy as np
import pickle
import os
import json
import collections
import DatabaseSQLite
import ModelInfo
import time


from typing import List, Dict, NoReturn, Union, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, normalize
from imblearn.over_sampling import ADASYN
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score

# JSONType = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONType = Union[str, bytes, bytearray]
RowAsDictType = Dict[str, Union[str, float, int]]

class ModelSGDClassifier:

    def __init__(self) -> None:
        self.model: SGDClassifier = None
        self.sc: StandardScaler = None
        self.df_original: pd.DataFrame = None
        self.last_sample_id: Optional[int] = None

    @staticmethod
    def read_csv_data(filepath: str, rows: int) -> pd.DataFrame:
        headers = ['Sale', 'SalesAmountInEuro', 'time_delay_for_conversion', 'click_timestamp', 'nb_clicks_1week',
                   'product_price', 'product_age_group', 'device_type', 'audience_id', 'product_gender',
                   'product_brand', 'product_category(1)', 'product_category(2)', 'product_category(3)',
                   'product_category(4)', 'product_category(5)', 'product_category(6)', 'product_category(7)',
                   'product_country', 'product_id', 'product_title', 'partner_id', 'user_id']
        df = pd.read_csv(
            filepath,
            sep='\t',
            nrows=rows,
            names=headers,
            low_memory=False,
            usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
        return df

    @staticmethod
    def create_one_hot_vectors(df: pd.DataFrame) -> pd.DataFrame:
        index_of_columns_to_change_to_one_hot_vectors = [6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 21]

        df_only_one_hot_vectors_columns = pd.get_dummies(df.iloc[:, index_of_columns_to_change_to_one_hot_vectors])
        df_common_columns = df.iloc[:, [0, 1, 2, 4, 5]]
        df = pd.concat([df_common_columns, df_only_one_hot_vectors_columns], axis=1, sort=True)

        return df

    @staticmethod
    def read_required_column_names() -> List[str]:
        required_column_name_file = open('required_column_names_list.txt', 'r')
        required_column_names_list = required_column_name_file.read().splitlines()
        return required_column_names_list

    @staticmethod
    def remove_nones_in_dict(dictionary: RowAsDictType):
        result = {}
        for k, v in dictionary.items():
            if v is None:
                result[k] = 0 #TODO zmienić aby wszystko w dataframeach i slownikach bylo str
            else:
                result[k] = v
        return result

    def transform_df_into_df_with_one_hot_vectors(self, df_to_transform: pd.DataFrame) -> pd.DataFrame:
        data_as_dict = json.loads(df_to_transform.T.to_json())

        samples = []
        for row_number, row_as_dict in data_as_dict.items():
            transformed_row_as_dict = self.create_dict_as_transformed_row_and_set_no_one_hot_vectors_columns(row_as_dict)
            transformed_row_as_dict = self.set_values_to_one_hot_vectors_columns(row_as_dict, transformed_row_as_dict)
            samples.append(transformed_row_as_dict)

        df = pd.DataFrame(samples)
        df = df.fillna(0)

        return df

    def set_values_to_one_hot_vectors_columns(self, old_dict: RowAsDictType, new_dict: RowAsDictType) -> RowAsDictType:
        required_column_names_list = self.read_required_column_names()

        for column_number, (column_name, cell_value) in enumerate(old_dict.items()):
            if column_number > 2:
                transformed_column_name = column_name + '_' + str(cell_value)
                if transformed_column_name in required_column_names_list:
                    new_dict[transformed_column_name] = 1


        # for column_name, value in row_as_dict.items():
        #     transformed_column_name = column_name + '_' + str(value)
        #     if transformed_column_name in required_column_names_list:
        #         transformed_row_as_dict[transformed_column_name] = 1

        new_dict = self.remove_nones_in_dict(new_dict)

        return new_dict

    def create_dict_as_transformed_row_and_set_no_one_hot_vectors_columns(self, old_dict: RowAsDictType) -> RowAsDictType:
        required_column_names_list = self.read_required_column_names()

        new_dict = dict.fromkeys(required_column_names_list)
        new_dict['Sale'] = old_dict['Sale']
        new_dict['SalesAmountInEuro'] = old_dict['SalesAmountInEuro']
        new_dict['time_delay_for_conversion'] = old_dict[
            'time_delay_for_conversion']
        new_dict['click_timestamp'] = old_dict['click_timestamp']
        new_dict['nb_clicks_1week'] = old_dict['nb_clicks_1week']

        return new_dict

    def create_model_and_save(self, json_training_data: JSONType) -> None:
        self.df_original = pd.read_json(json_training_data)
        # df_one_hot_vectors = df_one_hot_vectors.dropna(axis=0)  # usuwanie wierszy, które zawierają null

        start = time.time()
        x_test_std, x_train_std, y_test, y_train = self.create_train_and_test_sets()
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

    def create_train_and_test_sets(self) -> List[np.ndarray]:
        df_one_hot_vectors = self.transform_df_into_df_with_one_hot_vectors(self.df_original)
        x = df_one_hot_vectors.iloc[:, 3:].values
        y = df_one_hot_vectors.iloc[:, :1].values.ravel()  # tutaj powinny być chyba 3 kolumny
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
        # print('Liczba etykiet w zbiorze y:', np.bincount(y))
        # print('Liczba etykiet w zbiorze y_train:', np.bincount(y_train))
        # print('Liczba etykiet w zbiorze y_test:', np.bincount(y_test))

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

    def update_model(self) -> None:
        db = DatabaseSQLite.DatabaseSQLite()
        df_samples_to_update = db.get_samples_to_update_model()
        df_one_hot_vectors = self.transform_df_into_df_with_one_hot_vectors(df_samples_to_update)
        # df_one_hot_vectors = df_one_hot_vectors.dropna(axis=0)  # usuwanie wierszy, które zawierają null

        x = df_one_hot_vectors.iloc[:, 3:].values
        y = df_one_hot_vectors['Sale'].values.ravel()

        x = self.pca.transform(x)
        adasyn = ADASYN(random_state=1)
        x, y = adasyn.fit_resample(x, y) #TODO blad jak wszystkie probki sa z jedne klasy
        x = self.sc.transform(x)
        x = normalize(x, norm='l2')
        y = np.array([int(i) for i in y])

        self.model.partial_fit(x, y, classes=np.array([0, 1]))
        self.save_model()
        print("LOG: updating model DONE")

    def transform_json_row_in_one_hot_vectors_list_of_values(self, row_as_json: JSONType) -> List[Union[str, int, float]]:
        row_as_dict = json.loads(row_as_json)

        transformed_row_as_dict = self.create_dict_as_transformed_row_and_set_no_one_hot_vectors_columns(row_as_dict)
        transformed_row_as_dict = self.set_values_to_one_hot_vectors_columns(row_as_dict, transformed_row_as_dict)

        return list(transformed_row_as_dict.values())

    def predict(self, x: JSONType) -> Tuple[np.ndarray, np.ndarray]:
        transformed_x = self.transform_json_row_in_one_hot_vectors_list_of_values(x)
        transformed_x = transformed_x[3:]  # remove sales features from sample
        transformed_x = self.pca.transform([transformed_x])
        transformed_x = self.sc.transform(transformed_x)
        transformed_x = normalize(transformed_x, norm='l2')

        probability = self.model.predict_proba(transformed_x)
        y = self.model.predict(transformed_x)

        db = DatabaseSQLite.DatabaseSQLite()
        db.add_row_from_json(sample_json=x)

        return y, probability

    def save_model(self) -> None:
        if self.model is None:
            print("LOG: " + "There is not model available. Must be created.")
        # zapisywanie modelu do pliku
        # current_dir = os.path.dirname(__file__)
        # dest = os.path.join('pickle_objects')
        # if not os.path.exists(dest):
        #     os.makedirs(dest)
        # pickle.dump(self.model, open(os.path.join(dest, "SGDClassifier.pkl"), mode='wb'), protocol=4)
        # print("LOG: " + "model saved in directory: " + current_dir + '\\' + dest + '\SGDClassifier.pkl')

        db = DatabaseSQLite.DatabaseSQLite()
        last_sample_id = db.get_last_sample_id()
        model_info = ModelInfo.ModelInfo(None, "SGDClassifier", 0, None, last_sample_id, self.model, self.sc, self.pca)
        db.add_model(model_info)
        print("LOG: saving model DONE")

    def load_model(self) -> None:
        # current_dir = os.path.dirname(__file__)
        # self.model = pickle.load(open(os.path.join(current_dir, 'pickle_objects', 'SGDClassifier.pkl'), mode='rb'))
        # print("LOG: " + "model load from directory: " + current_dir + '\SGDClassifier.pkl')
        db = DatabaseSQLite.DatabaseSQLite()
        model_info = db.get_last_model_info()
        self.model = model_info.model
        self.sc = model_info.sc
        self.pca = model_info.pca

        print("LOG: " + "Model loaded.")

    def test(self, n_samples_for_training: int, n_samples_for_testing: int) -> None:
        n_samples_toread_from_csv = n_samples_for_training + n_samples_for_testing + 1
        df = pd.read_csv('/home/marcin/PycharmProjects/Engineering_Thesis/dataset/CriteoSearchData-sorted-no-duplicates.csv',
                         sep='\t',
                         nrows=n_samples_toread_from_csv,
                         low_memory=False)

        training_data_json = df[1:n_samples_for_training+1].to_json()

        print("Training...")
        self.create_model_and_save(training_data_json)
        print("DONE")
        print("Testing...")
        y_pred = np.array([])
        y_test = np.array([])

        for id, row in df[n_samples_for_training+1:n_samples_for_training+2+n_samples_for_testing].iterrows():
            y, prob = self.predict(row.to_json())
            y_test = np.append(y_test, row['Sale'])
            y_pred = np.append(y_pred, y)

        print(collections.Counter(y_test))
        print(collections.Counter(y_pred))

        print("LOG: testing model DONE")
        print('balanced_accuracy_score: {0}'.format(balanced_accuracy_score(y_test, y_pred)))
        print('accuracy_score: {0}'.format(accuracy_score(y_test, y_pred)))

        print('Nieprawidłowo sklasyfikowane próbki: %d' % (y_test != y_pred).sum())

        print('classification_report :\n', classification_report(y_test, y_pred))
        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        print(confmat)


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
    m.load_model()
    # print(help(m))
    # m.update_model()
    m.test(n_samples_for_training=100000, n_samples_for_testing=100)
