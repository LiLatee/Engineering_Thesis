import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class Model:
    @staticmethod
    def read_csv_data(self, filepath, rows):
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
    def create_one_hot_vectors(self, df):
        print("=========================Liczba unikalnych wartości w każdej kolumnie=========================")
        columns = list(df)
        count_of_unique_values_in_each_row = []
        for i, c in enumerate(columns):
            count = len(np.unique(df[c].values))
            print(str(i) + "\t" + str(count) + "\t" + c)
            count_of_unique_values_in_each_row.append(count)

        index_of_columns_with_lower_than_1000_unique_values = []
        excluded_indexes = [0, 1, 2, 3, 4, 5, 6]
        for id, count in enumerate(count_of_unique_values_in_each_row):
            if id not in excluded_indexes:
                if count < 1000:
                    index_of_columns_with_lower_than_1000_unique_values.append(id)

        df = pd.get_dummies(df.iloc[:, index_of_columns_with_lower_than_1000_unique_values])

        return df

    def create_model(self, rows, filepath):
        self.df_original = self.read_csv_data(self, filepath=filepath, rows=rows)

        print("=========================Liczba brakujących wartości w każdej kolumnie=========================")
        print(self.df_original.isnull().sum())
        df = self.df_original.dropna(axis=0) # usuwanie wierszy, które zawierają null - kolumna np. kolumna title posiada

        df = self.create_one_hot_vectors(self, df)


        X = df.iloc[:, 3:].values # pierwsze 3 kolumny to nasze y, my używamy tylko jednej z nich
        y = df.iloc[:, :1].values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        print('Liczba etykiet w zbiorze y:', np.bincount(y))
        print('Liczba etykiet w zbiorze y_train:', np.bincount(y_train))
        print('Liczba etykiet w zbiorze y_test:', np.bincount(y_test))


        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        lr = SGDClassifier(loss='log', verbose=0, n_jobs=-1, random_state=1)
        lr.fit(X_train_std, y_train)

        self.model = lr

        y_pred = lr.predict(X_test_std)
        print('Nieprawidłowo sklasyfikowane próbki: %d' % (y_test != y_pred).sum())
        print('Dokładność: %.2f' % lr.score(X_test_std, y_test))

    def update_model(self):
        pass

if __name__ == '__main__':
    m = Model()
    m.create_model(rows=10000, filepath='D:\Projekty\Engineering_Thesis\Dataset\Criteo_Conversion_Search\CriteoSearchData.csv')
