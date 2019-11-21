# how to open cqlsh:
# 1. docker exec -it <container_with_cassandra> bash
# 2. cqlsh

import json

from cassandra.cluster import Cluster, Session
from cqlengine import columns
from cqlengine.models import Model
from cqlengine.connection import setup
from typing import List, Dict, Union
from cassandra.query import dict_factory
from cassandra.cqlengine.management import sync_table

RowAsDictType = Dict[str, Union[str, float, int]]


class Sample(Model):
    id = columns.Integer(primary_key=True)
    # id = columns.TimeUUID(primary_key=True)
    sale = columns.Text()
    sales_amount_in_euro = columns.Text()
    time_delay_for_conversion = columns.Text()
    click_timestamp = columns.Text()
    nb_clicks_1week = columns.Text()
    product_price = columns.Text()
    product_age_group = columns.Text()
    device_type = columns.Text()
    audience_id = columns.Text()
    product_gender = columns.Text()
    product_brand = columns.Text()
    product_category_1 = columns.Text()
    product_category_2 = columns.Text()
    product_category_3 = columns.Text()
    product_category_4 = columns.Text()
    product_category_5 = columns.Text()
    product_category_6 = columns.Text()
    product_category_7 = columns.Text()
    product_country = columns.Text()
    product_id = columns.Text()
    product_title = columns.Text()
    partner_id = columns.Text()
    user_id = columns.Text()
    predicted = columns.Text()
    probabilities = columns.List(columns.Float)

import time
class CassandraClient:

    def __init__(self):
        super().__init__()
        self.KEYSPACE = 'keyspace_name'
        self.SAMPLE_TABLE = 'sample'
        self.MODEL_HISTORY_TABLE = 'model_history'
        self.LAST_SAMPLE_ID = 1

        is_exception = True
        while is_exception:
            try:
                self.setup_cassandra()
                self.session = self.get_session()
            except:
                is_exception = True
                continue
            is_exception = False

        self.create_tables()


    def setup_cassandra(self):
        # setup(hosts=['127.0.0.1'], default_keyspace=self.KEYSPACE)
        setup(hosts=['cassandra_api'], default_keyspace=self.KEYSPACE)

    def get_session(self):
        # cluster = Cluster(['127.0.0.1'], port=9042)
        cluster = Cluster(['cassandra_api'], port=9042)

        session = cluster.connect()
        session.row_factory = dict_factory
        self.create_keyspace(session)
        session.set_keyspace(self.KEYSPACE)
        return session

    def create_tables(self):
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


    def create_keyspace(self, session) :
        session.execute("""
        CREATE KEYSPACE IF NOT EXISTS """ + self.KEYSPACE + """
        WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '1' }
        """)

    def delete_all_samples(self):
        query = 'TRUNCATE sample'
        self.session.execute(query)

if __name__ == '__main__':
    db = CassandraClient()
    db.delete_all_samples()