import pandas as pd
import numpy as np
import pickle
import os
import json
import collections

from model import DatabaseSQLite
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, normalize


class model_SGDClassifier:

    def __init__(self) -> None:
        self.model = None
        self.sc = None
        self.last_sample_id = None

    @staticmethod
    def read_csv_data(filepath, rows):
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
    def create_one_hot_vectors(df):
        # print("=========================Liczba unikalnych wartości w każdej kolumnie=========================")
        # columns = list(df)
        # count_of_unique_values_in_each_row = []
        # for i, c in enumerate(columns):
        #     count = len(np.unique(df[c].values))
        #     print(str(i) + "\t" + str(count) + "\t" + c)
        #     count_of_unique_values_in_each_row.append(count)
        #
        # index_of_columns_with_lower_than_1000_unique_values = []
        # excluded_indexes = [0, 1, 2, 3, 4, 5, 6]
        # for id, count in enumerate(count_of_unique_values_in_each_row):
        #     if id not in excluded_indexes:
        #         if count < 1000:
        #             index_of_columns_with_lower_than_1000_unique_values.append(id)
        #
        # df = pd.get_dummies(df.iloc[:, index_of_columns_with_lower_than_1000_unique_values])

        index_of_columns_to_change_to_one_hot_vectors = [6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 21]

        df_only_one_hot_vectors_columns = pd.get_dummies(df.iloc[:, index_of_columns_to_change_to_one_hot_vectors])
        df_common_columns = df.iloc[:, [0, 1, 2, 4, 5]]
        df = pd.concat([df_common_columns, df_only_one_hot_vectors_columns], axis=1, sort=True)

        return df

    def transform_df_into_df_with_one_hot_vectors(self, df_to_transform: pd.DataFrame) -> pd.DataFrame:
        required_column_names_list = self.read_required_column_names()
        data_as_dict = json.loads(df_to_transform.T.to_json())

        samples = []
        for row_number in data_as_dict:
            transformed_row_as_dict = dict.fromkeys(required_column_names_list)
            transformed_row_as_dict['Sale'] = data_as_dict[str(row_number)]['Sale']
            transformed_row_as_dict['SalesAmountInEuro'] = data_as_dict[str(row_number)]['SalesAmountInEuro']
            transformed_row_as_dict['time_delay_for_conversion'] = data_as_dict[str(row_number)][
                'time_delay_for_conversion']
            transformed_row_as_dict['click_timestamp'] = data_as_dict[str(row_number)]['click_timestamp']
            transformed_row_as_dict['nb_clicks_1week'] = data_as_dict[str(row_number)]['nb_clicks_1week']
            for column_number, (column_name, cell_value) in enumerate(
                    zip(data_as_dict[row_number].keys(), data_as_dict[row_number].values())):
                if column_number > 2:
                    transformed_column_name = column_name + '_' + str(cell_value)
                    if transformed_column_name in required_column_names_list:
                        transformed_row_as_dict[transformed_column_name] = 1
            samples.append(transformed_row_as_dict)

        df = pd.DataFrame(samples)
        df = df.fillna(0)

        return df

    def create_model_2(self, json_data: str) -> None:
        self.df_original = pd.read_json(json_data)

        df_one_hot_vectors = self.transform_df_into_df_with_one_hot_vectors(self.df_original)
        X = df_one_hot_vectors.iloc[:, 3:].values
        y = df_one_hot_vectors.iloc[:, :1].values.ravel()  # tutaj powinny być chyba 3 kolumny
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        # print('Liczba etykiet w zbiorze y:', np.bincount(y))
        # print('Liczba etykiet w zbiorze y_train:', np.bincount(y_train))
        # print('Liczba etykiet w zbiorze y_test:', np.bincount(y_test))

        self.sc = StandardScaler()
        self.sc.fit(X_train)
        X_train_std = self.sc.transform(X_train)
        X_train_std = normalize(X_train_std, norm='l2')
        X_test_std = self.sc.transform(X_test)
        X_test_std = normalize(X_test_std, norm='l2')

        # ppn = Perceptron(eta0=0.1, random_state=1, n_jobs=-1)
        # ppn.fit(X_train_std, y_train)

        # lr = LogisticRegression(C=1000.0, random_state=1, solver="lbfgs", n_jobs=-1, verbose=1)
        # lr.fit(X_train_std, y_train)

        # X_train_std, X_val_std, y_train, y_val = train_test_split(X_train_std, y_train, test_size=0.2, random_state=1)
        lr = SGDClassifier(loss='log', verbose=0, n_jobs=-1, random_state=1, tol=1e-3, max_iter=1000, shuffle=True)
        lr.fit(X_train_std, y_train)

        print(collections.Counter(y_test))
        y_pred = lr.predict(X_test_std)
        print(collections.Counter(y_pred))

        print('Nieprawidłowo sklasyfikowane próbki: %d' % (y_test != y_pred).sum())
        print('Dokładność: %.2f' % lr.score(X_test_std, y_test))

        self.model = lr
        self.save_model()

    def create_model(self, rows, filepath):
        self.df_original = self.read_csv_data(filepath=filepath, rows=rows)
        print("=========================Liczba brakujących wartości w każdej kolumnie=========================")
        print(self.df_original.isnull().sum())
        df = self.df_original.dropna(axis=0) # usuwanie wierszy, które zawierają null - kolumna np. kolumna title posiada

        df = self.create_one_hot_vectors(df)


        X = df.iloc[:, 3:].values # pierwsze 3 kolumny to nasze y, my używamy tylko jednej z nich
        y = df.iloc[:, :1].values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        # print('Liczba etykiet w zbiorze y:', np.bincount(y))
        # print('Liczba etykiet w zbiorze y_train:', np.bincount(y_train))
        # print('Liczba etykiet w zbiorze y_test:', np.bincount(y_test))


        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_train_std = normalize(X_train_std, norm='l2')
        X_test_std = sc.transform(X_test)
        X_test_std = normalize(X_test_std, norm='l2')


        lr = SGDClassifier(loss='log', verbose=0, n_jobs=-1, random_state=1)
        lr.fit(X_train_std, y_train)

        self.model = lr

        y_pred = lr.predict(X_test_std)
        print('Nieprawidłowo sklasyfikowane próbki: %d' % (y_test != y_pred).sum())
        print('Dokładność: %.2f' % lr.score(X_test_std, y_test))

    def update_model(self) -> None:
        db = DatabaseSQLite.DatabaseSQLite()
        df_samples_to_update = db.get_samples_to_update_model()
        df_one_hot_vectors = self.transform_df_into_df_with_one_hot_vectors(df_samples_to_update)

        x = df_one_hot_vectors.iloc[:, 3:].values
        y = df_one_hot_vectors['Sale'].values.ravel()
        print(df_one_hot_vectors)
        x = self.sc.transform(x)
        x = normalize(x, norm='l2')
        y = np.array([int(i) for i in y])

        self.model.partial_fit(x, y, classes=np.array([0, 1]))
        self.save_model()
        print("LOG: updating model DONE")

    def read_required_column_names(self) -> list:
        required_column_name_file = open('required_column_names_list.txt', 'r')
        required_column_names_list = required_column_name_file.read().splitlines()
        return required_column_names_list

    def remove_nones_in_dict(self, dictionary: dict):
        result = {}
        for k, v in dictionary.items():
            if v is None:
                result[k] = 0
            else:
                result[k] = v
        return result

    def transform_one_row_in_one_hot_vectors_row(self, row_as_json: json) -> list:
        required_column_names_list = self.read_required_column_names()

        row_as_dict = json.loads(row_as_json)
        transformed_row_as_dict = dict.fromkeys(required_column_names_list)

        transformed_row_as_dict['Sale'] = row_as_dict['Sale']
        transformed_row_as_dict['SalesAmountInEuro'] = row_as_dict['SalesAmountInEuro']
        transformed_row_as_dict['time_delay_for_conversion'] = row_as_dict['time_delay_for_conversion']
        transformed_row_as_dict['click_timestamp'] = int(row_as_dict['click_timestamp'])
        transformed_row_as_dict['nb_clicks_1week'] = int(row_as_dict['nb_clicks_1week'])

        for column_name, value in row_as_dict.items():
            transformed_column_name = column_name + '_' + str(value)
            if transformed_column_name in required_column_names_list:
                transformed_row_as_dict[transformed_column_name] = 1

        transformed_row_as_dict = self.remove_nones_in_dict(transformed_row_as_dict)

        return list(transformed_row_as_dict.values())

    def predict(self, x: json):
        transformed_x = self.transform_one_row_in_one_hot_vectors_row(x)
        transformed_x = transformed_x[3:]  # remove sales features from sample
        transformed_x = self.sc.transform([transformed_x])
        transformed_x = normalize(transformed_x, norm='l2')

        probability = self.model.predict_proba(transformed_x)
        y = self.model.predict(transformed_x)

        db = DatabaseSQLite.DatabaseSQLite()
        db.add_row_from_json(sample_json=x)

        return y, probability

    def save_model(self):
        if self.model is None:
            print("LOG: " + "model is not created")
            return
        # zapisywanie modelu do pliku
        # current_dir = os.path.dirname(__file__)
        # dest = os.path.join('pickle_objects')
        # if not os.path.exists(dest):
        #     os.makedirs(dest)
        # pickle.dump(self.model, open(os.path.join(dest, "SGDClassifier.pkl"), mode='wb'), protocol=4)
        # print("LOG: " + "model saved in directory: " + current_dir + '\\' + dest + '\SGDClassifier.pkl')


        db = DatabaseSQLite.DatabaseSQLite()
        model_binary = pickle.dumps(self.model)
        standard_scaler_binary = pickle.dumps(self.sc)
        last_sample_id = db.get_last_sample_id()
        db.add_model("SGDClassifier", 0, last_sample_id, model_binary, standard_scaler_binary)


    def load_model(self):
        # current_dir = os.path.dirname(__file__)
        # self.model = pickle.load(open(os.path.join(current_dir, 'pickle_objects', 'SGDClassifier.pkl'), mode='rb'))
        # print("LOG: " + "model load from directory: " + current_dir + '\SGDClassifier.pkl')
        db = DatabaseSQLite.DatabaseSQLite()
        self.model, self.sc, _ = db.get_last_model()

    def test(self, n_samples_for_training: int, n_samples_for_testing: int) -> None:
        n_samples_toread_from_csv = n_samples_for_training + n_samples_for_testing + 1
        headers = ['Sale', 'SalesAmountInEuro', 'time_delay_for_conversion', 'click_timestamp', 'nb_clicks_1week',
                   'product_price', 'product_age_group', 'device_type', 'audience_id', 'product_gender',
                   'product_brand', 'product_category(1)', 'product_category(2)', 'product_category(3)',
                   'product_category(4)', 'product_category(5)', 'product_category(6)', 'product_category(7)',
                   'product_country', 'product_id', 'product_title', 'partner_id', 'user_id']
        df = pd.read_csv('D:\Projekty\Engineering_Thesis\Dataset\Criteo_Conversion_Search\CriteoSearchData-sorted.csv',
                         sep='\t',
                         nrows=n_samples_toread_from_csv,
                         names=headers,
                         low_memory=False,
                         usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

        training_data_json = df[1:n_samples_for_training+1].to_json()

        # print("Training...")
        # self.create_model_2(training_data_json)
        # print("DONE")
        print("Testing...")
        f = np.array([])
        for id, row in df[n_samples_for_training+1:n_samples_for_training+2+n_samples_for_testing].iterrows():
            y, prob = self.predict(row.to_json())
            f = np.append(f, y)

        print(collections.Counter(f))



        # tests = []
        # for id, row in df[1001:2000].iterrows():
        #     transformed_x = self.transform_one_row_in_one_hot_vectors_row(row.to_json())
        #     transformed_x = transformed_x[3:]  # remove sales features from sample
        #     tests.append(transformed_x)
        # tests = np.asarray(tests)
        # tests = self.sc.transform(tests)
        # y = self.model.predict(tests)
        # print("y: %s\t probability: %s" % (str(y), str(444)))


if __name__ == '__main__':
    m = model_SGDClassifier()
    m.load_model()
    m.update_model()
    # m.test(1000, 5)
