import os
import pandas as pd
import sqlite3
import json
import pickle
import model_info
from sqlite3 import Error
from typing import Union, Dict, List
from model_info import ModelInfo

# JSONType = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONType = Union[str, bytes, bytearray]
RowAsDictType = Dict[str, Union[str, float, int]]


# TODO: id modelu przy wyszukiwaniu zmienić ze stałej
class DatabaseSQLite:
    def __init__(self):
        self.db_file_samples = '/data_samples/samples.db' # dla systemu: '/data_samples/samples.db', dla testów: 'data_samples/samples.db'
        self.db_file_models = '/data_models/models.db'

        if os.path.exists(self.db_file_samples):
            os.remove(self.db_file_samples)

        if os.path.exists(self.db_file_models):
            os.remove(self.db_file_models)

        self.create_tables()

    def create_connection_samples(self) -> sqlite3.Connection:
        def dict_factory(cursor, row):
            d = {}
            for idx, col in enumerate(cursor.description):
                d[col[0]] = row[idx]
            return d

        try:
            conn = sqlite3.connect(self.db_file_samples, timeout=60)
            conn.row_factory = dict_factory
            return conn
        except Error as e:
            print(e)

    def create_connection_models(self) -> sqlite3.Connection:
        def dict_factory(cursor, row):
            d = {}
            for idx, col in enumerate(cursor.description):
                d[col[0]] = row[idx]
            return d

        try:
            conn = sqlite3.connect(self.db_file_models)
            conn.row_factory = dict_factory
            return conn
        except Error as e:
            print(e)

    def create_tables(self) -> None:
        sql_create_samples_table = """ CREATE TABLE IF NOT EXISTS samples (
                                            id INTEGER PRIMARY KEY,
                                            sale TEXT,
                                            sales_amount_in_euro TEXT,
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
                                            user_id TEXT,
                                            predicted TEXT,
                                            probabilities TEXT
                                            ); """

        sql_create_models_table = """ CREATE TABLE IF NOT EXISTS models (
                                        id INTEGER PRIMARY KEY,
                                        name TEXT,
                                        version INTEGER,
                                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                                        last_sample_id TEXT,
                                        model BLOB,
                                        standard_scaler BLOB,
                                        df_product_clicks_views BLOB
                                        );
        """

        conn = self.create_connection_samples()
        try:
            c = conn.cursor()
            c.execute(sql_create_samples_table)
        except Error as e:
            print(e)
        conn.commit()
        conn.close()

        conn = self.create_connection_models()
        try:
            c = conn.cursor()
            c.execute(sql_create_models_table)
        except Error as e:
            print(e)
        conn.commit()
        conn.close()

    def insert_sample_as_dict(self, sample_dict: RowAsDictType) -> None:
        sql = ''' INSERT INTO samples(  sale,
                                        sales_amount_in_euro,
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
                                        user_id,
                                        predicted,
                                        probabilities)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''

        sample_array = sample_dict.values()
        conn = self.create_connection_samples()

        cur = conn.cursor()
        cur.execute(sql, tuple(sample_array))
        conn.commit()
        conn.close()

        return cur.lastrowid

    def insert_ModelInfo(self, model_info: model_info) -> None:
        sql_query = """ INSERT INTO models (name, version, last_sample_id, model, standard_scaler, df_product_clicks_views) VALUES (?,?,?,?,?,?)"""
        binary_model = pickle.dumps(model_info.model)
        binary_standard_scaler = pickle.dumps(model_info.standard_scaler)
        binary_df_product_clicks_views = pickle.dumps(model_info.df_product_clicks_views)

        conn = self.create_connection_models()
        cur = conn.cursor()
        cur.execute(sql_query,
                    (model_info.name,
                     model_info.version,
                     str(model_info.last_sample_id),
                     sqlite3.Binary(binary_model),
                     sqlite3.Binary(binary_standard_scaler),
                     sqlite3.Binary(binary_df_product_clicks_views))
                    )

        conn.commit()
        conn.close()

    def get_last_ModelInfo(self) -> model_info:
        sql_query = """ SELECT * FROM models
                        WHERE id = (SELECT max(id) FROM models)"""
        conn = self.create_connection_models()
        conn.row_factory = None
        result = conn.execute(sql_query).fetchone()
        if result is None:
            return None
        model_info = ModelInfo()
        model_info.id = result[0]
        model_info.name = result[1]
        model_info.version = result[2]
        model_info.date_of_create = result[3]
        model_info.last_sample_id = result[4]
        model_info.model = pickle.loads(result[5])
        model_info.standard_scaler = pickle.loads(result[6])
        model_info.df_product_clicks_views = pickle.loads(result[7])

        conn.commit()
        conn.close()
        return model_info

    def get_last_sample_id(self) -> int:
        conn = self.create_connection_samples()
        sql_query = """SELECT id from samples WHERE id=(SELECT max(id) FROM samples) """
        result = conn.execute(sql_query).fetchone()
        if result is None:
            return -1
        conn.commit()
        conn.close()
        return result[0]

    def get_last_version_of_specified_model(self, model_name: str) -> int:
        conn = self.create_connection_models()
        sql_query = 'SELECT version from models WHERE timestamp = (SELECT max(timestamp) FROM models WHERE name = "' + model_name + '")'
        result = conn.execute(sql_query).fetchone()
        conn.commit() #todo chyba można wywalić w wielu miejscach
        conn.close()
        if result is None:
            return -1
        else:
            return int(result['version'])

    def get_id_of_last_specified_model(self, model_name: str) -> int:
        conn = self.create_connection_models()
        sql_query = 'SELECT id from models WHERE timestamp = (SELECT max(timestamp) FROM models WHERE name = "' + model_name + '")'
        result = conn.execute(sql_query).fetchone()
        conn.commit() #todo chyba można wywalić w wielu miejscach
        conn.close()
        if result is None:
            return -1
        else:
            return int(result['id'])

    def get_samples_to_update_model_as_list_of_dicts(self, last_sample_id) -> List[RowAsDictType]:
        conn = self.create_connection_samples()

        cur = conn.cursor()
        cur.execute('SELECT * FROM samples WHERE id > (?)', (str(last_sample_id),))
        list_of_dicts = cur.fetchall()
        conn.commit()
        conn.close()
        for sample in list_of_dicts:  #todo to samo z predicted z str na int? zmienić w bazie
            sample['probabilities'] = json.loads(sample['probabilities'])
        return list_of_dicts

    def get_all_models_history_as_list_of_dicts(self) -> List[RowAsDictType]:
        conn = self.create_connection_models()
        # df = pd.read_sql_query('SELECT * FROM models', conn)
        # return df

        cur = conn.cursor()
        cur.execute('SELECT * FROM models')
        list_of_dicts = cur.fetchall()
        conn.commit()
        conn.close()
        return list_of_dicts

    def get_all_models_history_as_list_of_ModelInfo(self) -> List[ModelInfo]:
        conn = self.create_connection_models()
        # df = pd.read_sql_query('SELECT * FROM models', conn)
        # return df

        cur = conn.cursor()
        cur.execute('SELECT * FROM models')
        list_of_dicts = cur.fetchall()
        conn.commit()
        conn.close()
        list_of_ModelInfo = []
        for x in list_of_dicts:
            model_info = ModelInfo()
            list_of_values = list(x.values())
            model_info.id = list_of_values[0]
            model_info.name = list_of_values[1]
            model_info.version = list_of_values[2]
            model_info.date_of_create = list_of_values[3]
            model_info.last_sample_id = list_of_values[4]
            model_info.model = pickle.loads(list_of_values[5])
            model_info.standard_scaler = pickle.loads(list_of_values[6])
            model_info.df_product_clicks_views = pickle.loads(list_of_values[7])

            list_of_ModelInfo.append(model_info)

        return list_of_ModelInfo

    def get_all_samples_as_list_of_dicts(self) -> List[RowAsDictType]:
        conn = self.create_connection_samples()
        # df = pd.read_sql_query('SELECT * FROM samples', conn)
        # conn.commit()
        # conn.close()
        # return df

        cur = conn.cursor()
        cur.execute('SELECT * FROM samples')
        list_of_dicts = cur.fetchall()
        conn.commit()
        conn.close()
        for sample in list_of_dicts:#todo to samo z predicted z str na int? zmienić w bazie
            sample['probabilities'] = json.loads(sample['probabilities'])
        return list_of_dicts


    @staticmethod
    def read_csv_data(filepath, rows):
        headers = ['sale', 'sales_amount_in_euro', 'time_delay_for_conversion', 'click_timestamp', 'nb_clicks_1week',
                   'product_price', 'product_age_group', 'device_type', 'audience_id', 'product_gender',
                   'product_brand', 'product_category_1', 'product_category_2', 'product_category_3',
                   'product_category_4', 'product_category_5', 'product_category_6', 'product_category_7',
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
    result = db.get_last_version_of_specified_model('SGDClassifier')
    print(result)
    # db.create_tables()
    # df = db.read_csv_data(
    #     filepath='/home/marcin/PycharmProjects/Engineering_Thesis/data/CriteoSearchData-sorted.csv', rows=10)
    # one_row = df[1:2].squeeze()
    # db.add_row_from_json(one_row.to_json())
    # print(db.select_all_samples_as_df())
    # print(db.get_all_models_history_as_df()['last_sample_id'])
