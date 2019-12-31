# how to open cqlsh:
# 1. docker exec -it <container_with_cassandra> bash
# 2. cqlsh

import json

import uuid
from cassandra.cluster import Cluster, Session
from typing import List, Dict, Union
from cassandra.query import dict_factory

from cqlengine.connection import setup

RowAsDictType = Dict[str, Union[str, float, int]]


class CassandraClient:

    def __init__(self, number_of_model=1) -> None:
        super().__init__()
        self.KEYSPACE = 'keyspace_name'
        self.TABLE_NAME = 'model_' + str(number_of_model)

        cluster = Cluster(['cassandra_api'], port=9042)
        # cluster = Cluster(['127.0.0.1'], port=9042)
        self.session = cluster.connect()
        self.session.row_factory = dict_factory
        self.create_keyspace(self.session)
        self.session.set_keyspace(self.KEYSPACE)
        setup(hosts=['cassandra_api'], default_keyspace=self.KEYSPACE)
        # setup(hosts=['127.0.0.1'], default_keyspace=self.KEYSPACE)

        self.create_samples_table()

    def restart_cassandra(self) -> None:
        # self.setup_cassandra()
        self.session.execute('DROP KEYSPACE IF EXISTS ' + self.KEYSPACE)
        self.create_keyspace(self.session)

    def create_samples_table(self) -> None:
        query_create_samples_table = " CREATE TABLE IF NOT EXISTS " + self.TABLE_NAME + """ (
                                            id TimeUUID PRIMARY KEY,
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
                                            probabilities LIST<FLOAT>
                                            ); """
        query_result = self.session.execute(query_create_samples_table)

    def create_tables(self) -> None:
        query_create_samples_table = """ CREATE TABLE IF NOT EXISTS sample (
                                            id INT PRIMARY KEY,
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
                                            probabilities LIST<FLOAT>
                                            ); """
        query_result = self.session.execute(query_create_samples_table)

    def create_keyspace(self, session: Session) -> None:
        session.execute("""
        CREATE KEYSPACE IF NOT EXISTS """ + self.KEYSPACE + """
        WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '1' }
        """)

    def insert_sample(self, sample: RowAsDictType) -> None:
        query = "INSERT INTO " + self.TABLE_NAME + """ ( 
                                 id,
                                 sale,
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
                                 probabilities
                             ) 
                             VALUES (
                                  %(id)s,
                                  %(sale)s,
                                  %(sales_amount_in_euro)s,
                                  %(time_delay_for_conversion)s,
                                  %(click_timestamp)s,
                                  %(nb_clicks_1week)s,
                                  %(product_price)s,
                                  %(product_age_group)s,
                                  %(device_type)s,
                                  %(audience_id)s,
                                  %(product_gender)s,
                                  %(product_brand)s,
                                  %(product_category_1)s,
                                  %(product_category_2)s,
                                  %(product_category_3)s,
                                  %(product_category_4)s,
                                  %(product_category_5)s,
                                  %(product_category_6)s,
                                  %(product_category_7)s,
                                  %(product_country)s,
                                  %(product_id)s,
                                  %(product_title)s,
                                  %(partner_id)s,
                                  %(user_id)s,
                                  %(predicted)s,
                                  %(probabilities)s
                                  )"""
        variables = {
                     'id': uuid.uuid1(),
                     'sale': str(sample.get("sale")),
                     'sales_amount_in_euro': str(sample.get("sales_amount_in_euro")),
                     'time_delay_for_conversion': str(sample.get("time_delay_for_conversion")),
                     'click_timestamp': str(sample.get("click_timestamp")),
                     'nb_clicks_1week': str(sample.get("nb_clicks_1week")),
                     'product_price': str(sample.get("product_price")),
                     'product_age_group': str(sample.get("product_age_group")),
                     'device_type': str(sample.get("device_type")),
                     'audience_id': str(sample.get("audience_id")),
                     'product_gender': str(sample.get("product_gender")),
                     'product_brand': str(sample.get("product_brand")),
                     'product_category_1': str(sample.get("product_category_1")),
                     'product_category_2': str(sample.get("product_category_2")),
                     'product_category_3': str(sample.get("product_category_3")),
                     'product_category_4': str(sample.get("product_category_4")),
                     'product_category_5': str(sample.get("product_category_5")),
                     'product_category_6': str(sample.get("product_category_6")),
                     'product_category_7': str(sample.get("product_category_7")),
                     'product_country': str(sample.get("product_country")),
                     'product_id': str(sample.get("product_id")),
                     'product_title': str(sample.get("product_title")),
                     'partner_id': str(sample.get("partner_id")),
                     'user_id': str(sample.get("user_id")),
                     'predicted': str(sample.get("predicted")),
                     'probabilities': json.loads(sample.get("probabilities"))
                    }
        self.session.execute(query, variables)


    def add_some_data(self) -> None:
        sample = {
            "id": -12,
            "sale": "888",
            "sales_amount_in_euro": "1",
            "time_delay_for_conversion": "2121",
            "click_timestamp": "1213123",
            "nb_clicks_1week": "12",
            "product_price": "321.3",
            "product_age_group": "2",
            "device_type": "312",
            "audience_id": "321",
            "product_gender": "2",
            "product_brand": "dewe21",
            "product_category_1": "aaa",
            "product_category_2": "bb",
            "product_category_3": "ccc",
            "product_category_4": "ddd",
            "product_category_5": "ee",
            "product_category_6": "ff",
            "product_category_7": "gg",
            "product_country": "Asdg",
            "product_id": "234",
            "product_title": "TTT",
            "partner_id": "31",
            "user_id": "35",
            "predicted": "1",
            "probabilities": json.dumps([0.6, 0.4])
        }

        self.insert_sample(sample)

    # todo dodać obsług≥e błedów, jak select nic nie zwraca WSZEDZIE
    def get_last_sample_id(self):
        query_result = self.session.execute("SELECT MAX(id) FROM " + str(self.TABLE_NAME))
        result = []
        for sample in query_result:
            result.append(sample)

        return result[0]['system.max(id)']

    def get_all_samples_as_list_of_dicts(self) -> List[dict]:
        query = "SELECT * FROM " + str(self.TABLE_NAME)
        query_result = self.session.execute(query)

        result = []
        for sample in query_result:
            result.append(sample)

        return result

    def delete_all_samples(self):
        self.session.execute("TRUNCATE " + str(self.TABLE_NAME))

    def get_samples_for_update_model_as_list_of_dicts(self, id=None):
        if id is None:
            query = "SELECT * FROM " + str(self.TABLE_NAME)
            query_result = self.session.execute(query)
        else:
            query = "SELECT * FROM " + str(self.TABLE_NAME) + " WHERE id > " + str(id) + " ALLOW FILTERING"
            query_result = self.session.execute(query)

        result = []
        for sample in query_result:
            sample.pop('id', None)
            sample.pop('predicted', None)
            sample.pop('probabilities', None)
            result.append(sample)

        return result


if __name__ == '__main__':
    db = CassandraClient()
    # db.restart_cassandra()
    # db.add_some_data()
    print(len(db.get_samples_for_update_model_as_list_of_dicts()))
    id = db.get_last_sample_id()
    db.add_some_data()
    print(len(db.get_samples_for_update_model_as_list_of_dicts(id)))
    print(len(db.get_samples_for_update_model_as_list_of_dicts()))
    # id = db.get_last_sample_id()
    # print(db.get_samples_for_update_model_as_list_of_dicts(id))
    # print(type(db.get_samples_for_update_model_as_list_of_dicts(id)[0]))
    # print(len(db.get_samples_for_update_model_as_list_of_dicts(id)))
