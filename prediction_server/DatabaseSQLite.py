import pandas as pd
import sqlite3
import json
import pickle
import ModelInfo
import time
from sqlite3 import Error
from typing import Union, Dict, Any, List

# JSONType = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONType = Union[str, bytes, bytearray]


# TODO: id modelu przy wyszukiwaniu zmienić ze stałej
class DatabaseSQLite:
    def __init__(self):
        self.create_tables()

    def __create_connection(self, db_file: str) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(db_file)
            return conn
        except Error as e:
            print(e)
        return None

    def create_tables(self) -> None:
        sql_create_samples_table = """ CREATE TABLE IF NOT EXISTS samples (
                                            id INTEGER PRIMARY KEY,
                                            Sale TEXT,
                                            SalesAmountInEuro TEXT,
                                            time_delay_for_conversion TEXT,
                                            click_timestamp TEXT,
                                            nb_clicks_1week TEXT,
                                            product_price TEXT,
                                            product_age_group TEXT,
                                            device_type TEXT,
                                            audience_id TEXT,
                                            product_gender TEXT,
                                            product_brand TEXT,
                                            product_category_1 TEXT,
                                            product_category_2 TEXT,
                                            product_category_3 TEXT,
                                            product_category_4 TEXT,
                                            product_category_5 TEXT,
                                            product_category_6 TEXT,
                                            product_category_7 TEXT,
                                            product_country TEXT,
                                            product_id TEXT,
                                            product_title TEXT,
                                            partner_id TEXT,
                                            user_id TEXT
                                            ); """
        sql_create_model_history_table = """ CREATE TABLE IF NOT EXISTS model_history (
        id INTEGER PRIMARY KEY,
        name TEXT,
        version INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_sample_id INTEGER,
        model BLOB,
        standard_scaler BLOB,
        pca BLOB
        );
        """

        conn = self.__create_connection('sqlite3.db')
        try:
            c = conn.cursor()
            c.execute(sql_create_samples_table)
            c.execute(sql_create_model_history_table)
        except Error as e:
            print(e)
        conn.commit()
        conn.close()

    def add_row_from_json(self, sample_json: JSONType) -> None:
        sql = ''' INSERT INTO samples(  Sale,
                                        SalesAmountInEuro,
                                        time_delay_for_conversion,
                                        click_timestamp,
                                        nb_clicks_1week,
                                        product_price,
                                        product_age_group,
                                        device_type,
                                        audience_id,
                                        product_gender,
                                        product_brand,
                                        product_category_1,
                                        product_category_2,
                                        product_category_3,
                                        product_category_4,
                                        product_category_5,
                                        product_category_6,
                                        product_category_7,
                                        product_country,
                                        product_id,
                                        product_title,
                                        partner_id,
                                        user_id )
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''

        sample_dict = json.loads(sample_json)
        sample_array = sample_dict.values()

        conn = self.__create_connection('sqlite3.db')

        cur = conn.cursor()
        cur.execute(sql, tuple(sample_array))
        conn.commit()
        conn.close()

        return cur.lastrowid

    def add_model(self, model_info: ModelInfo) -> None:
        sql_query = """ INSERT INTO model_history (name, version, last_sample_id, model, standard_scaler, pca) VALUES (?,?,?,?,?,?)"""
        binary_model = pickle.dumps(model_info.model)
        binary_standard_scaler = pickle.dumps(model_info.sc)
        binary_pca = pickle.dumps(model_info.pca)

        conn = self.__create_connection('sqlite3.db')
        cur = conn.cursor()
        cur.execute(sql_query,
                    (model_info.name,
                     model_info.version,
                     model_info.last_sample_id,
                     sqlite3.Binary(binary_model),
                     sqlite3.Binary(binary_standard_scaler),
                     sqlite3.Binary(binary_pca))
                    )
        conn.commit()
        conn.close()

    def get_last_model_info(self) -> ModelInfo:
        sql_query = """ SELECT * FROM model_history
                        WHERE id = (SELECT max(id) FROM model_history)"""
        conn = self.__create_connection('sqlite3.db')
        result = conn.execute(sql_query).fetchone()
        model_info = ModelInfo.ModelInfo(result[0], result[1], result[2], result[3], result[4],
                                         pickle.loads(result[5]), pickle.loads(result[6]), pickle.loads((result[7])))

        # model = pickle.loads(result[0])
        # standard_scaler = pickle.loads(result[1])
        # last_sample_id = result[2]

        return model_info

    def get_last_sample_id(self) -> int:
        conn = self.__create_connection('sqlite3.db')
        sql_query = """SELECT id from samples WHERE id=(SELECT max(id) FROM samples) """
        result = conn.execute(sql_query).fetchone()
        if result is None:
            return -1
        return result[0]

    def get_samples_to_update_model_as_df(self) -> pd.DataFrame:
        conn = self.__create_connection('sqlite3.db')
        last_sample_id = self.get_last_model_info().last_sample_id
        df = pd.read_sql_query('SELECT * FROM samples WHERE id >' + str(last_sample_id), conn)
        return df

    def get_all_models_history_as_df(self) -> pd.DataFrame:
        conn = self.__create_connection('sqlite3.db')
        df = pd.read_sql_query('SELECT * FROM model_history', conn)
        return df

    def get_all_samples_as_df(self) -> pd.DataFrame:
        conn = self.__create_connection('sqlite3.db')
        df = pd.read_sql_query('SELECT * FROM samples', conn)
        conn.commit()
        conn.close()
        return df

        # cur = conn.cursor()
        # cur.execute("SELECT * FROM samples")
        #
        # rows = cur.fetchall()
        # for row in rows:
        #     print(row)
        #

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


if __name__ == '__main__':
    db = DatabaseSQLite()
    # db.create_tables()
    df = db.read_csv_data(
        filepath='/home/marcin/PycharmProjects/Engineering_Thesis/dataset/CriteoSearchData-sorted.csv', rows=10)
    one_row = df[1:2].squeeze()
    # db.add_row_from_json(one_row.to_json())
    # print(db.select_all_samples_as_df())
    # print(db.get_all_models_history_as_df()['last_sample_id'])
