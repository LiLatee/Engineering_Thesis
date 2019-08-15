import pandas as pd
import sqlite3
import json
import pickle

from sqlite3 import Error


#TODO: id modelu przy wyszukiwaniu zmienić ze stałej
class DatabaseSQLite:

    @staticmethod
    def __init__(self):
        DatabaseSQLite.create_tables()

    @staticmethod
    def __create_connection():
        """ create a database connection to a SQLite database """
        db_file_name = 'database.db'
        try:
            conn = sqlite3.connect(db_file_name)
            return conn
        except Error as e:
            print(e)
        return None

    @staticmethod
    def create_tables():
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
        standard_scaler BLOB
        );
        """

        conn = DatabaseSQLite.__create_connection()
        try:
            c = conn.cursor()
            c.execute(sql_create_samples_table)
            c.execute(sql_create_model_history_table)
        except Error as e:
            print(e)
        conn.commit()
        conn.close()

    @staticmethod
    def add_row_from_json(sample_json: json):
        DatabaseSQLite.create_tables()
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

        conn = DatabaseSQLite.__create_connection()
        cur = conn.cursor()
        cur.execute(sql, tuple(sample_array))
        conn.commit()
        conn.close()
        return cur.lastrowid
    
    @staticmethod
    def add_model(name, version, last_sample_id, binary_model, binary_standard_scaler):
        DatabaseSQLite.create_tables()
        sql_query = """ INSERT INTO model_history (name, version, last_sample_id, model, standard_scaler) VALUES (?,?,?,?,?)"""
        conn = DatabaseSQLite.__create_connection()
        cur = conn.cursor()
        cur.execute(sql_query, (name, version, last_sample_id, sqlite3.Binary(binary_model), sqlite3.Binary(binary_standard_scaler)))
        conn.commit()
        conn.close()

    @staticmethod
    def get_last_model():
        sql_query = """ SELECT model, standard_scaler, last_sample_id FROM model_history
                        WHERE id = (SELECT max(id) FROM model_history)"""
        conn = DatabaseSQLite.__create_connection()
        result = conn.execute(sql_query).fetchone()
        model = pickle.loads(result[0])
        standard_scaler = pickle.loads(result[1])
        last_sample_id = result[2]

        return model, standard_scaler, last_sample_id

    @staticmethod
    def get_last_sample_id() -> int:
        conn = DatabaseSQLite.__create_connection()
        sql_query = """SELECT id from samples WHERE id=(SELECT max(id) FROM samples) """
        result = conn.execute(sql_query).fetchone()
        if result is None:
            return -1
        return result[0]

    @staticmethod
    def get_samples_to_update_model() -> pd.DataFrame:
        conn = DatabaseSQLite.__create_connection()
        _ , _ , last_sample_id = DatabaseSQLite.get_last_model()
        df = pd.read_sql_query('SELECT * FROM samples WHERE id >' + str(last_sample_id), conn)
        return df

    @staticmethod
    def get_all_models_history_as_df() -> pd.DataFrame:
        conn = DatabaseSQLite.__create_connection()
        df = pd.read_sql_query('SELECT * FROM model_history', conn)
        return df

    @staticmethod
    def get_all_samples_as_df() -> pd.DataFrame:
        conn = DatabaseSQLite.__create_connection()
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
    # db = DatabaseSQLite()
    # db.create_tables()
    df = DatabaseSQLite.read_csv_data(
            filepath='/home/marcin/PycharmProjects/Engineering_Thesis/dataset/CriteoSearchData-sorted.csv', rows=10)
    one_row = df[1:2].squeeze()
    # db.add_row_from_json(one_row.to_json())
    # print(db.select_all_samples_as_df())
    # print(db.get_all_models_history_as_df()['last_sample_id'])

