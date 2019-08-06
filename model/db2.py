import sqlalchemy as db
import pandas as pd
import sqlite3
import time

from sqlite3 import Error

class db2:

    def __create_connection(self, db_file):
        """ create a database connection to a SQLite database """
        try:
            conn = sqlite3.connect(db_file)
            return conn
        except Error as e:
            print(e)
        return None

    def create_table(self):
        # """ create a table from the create_table_sql statement
        # :param conn: Connection object
        # :param create_table_sql: a CREATE TABLE statement
        # :return:
        # """

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

        conn = self.__create_connection('samples.db')
        try:
            c = conn.cursor()
            c.execute(sql_create_samples_table)
        except Error as e:
            print(e)
        conn.commit()
        conn.close()

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

    def add_row_from_json(self):
        df_test = self.read_csv_data(
            filepath='D:\Projekty\Engineering_Thesis\Dataset\Criteo_Conversion_Search\CriteoSearchData.csv', rows=10)

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

        conn = self.__create_connection('samples.db')
        cur = conn.cursor()
        for id, row in df_test.iterrows():
            cur.execute(sql, tuple(row.to_list()))

        conn.commit()
        conn.close()
        return cur.lastrowid

    def select_all(self):
        """
        Query all rows in the tasks table
        :param conn: the Connection object
        :return:
        """
        conn = self.__create_connection('samples.db')
        cur = conn.cursor()
        cur.execute("SELECT * FROM samples")

        rows = cur.fetchall()
        for row in rows:
            print(row)

        conn.commit()
        conn.close()


if __name__ == '__main__':
    db = db2()
    # db.create_table()
    # db.add_row()
    # db.add_row()
    db.select_all()

