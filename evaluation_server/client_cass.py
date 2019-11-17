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


class CassandraClient:

    def __init__(self) -> None:
        super().__init__()
        self.KEYSPACE = 'keyspace_name'
        self.SAMPLE_TABLE = 'sample'
        self.MODEL_HISTORY_TABLE = 'model_history'
        self.LAST_SAMPLE_ID = 1

        self.setup_cassandra()
        self.session = self.get_session()
        # self.restart_cassandra()
        self.create_tables()


    def setup_cassandra(self) -> None:
        # setup(hosts=['127.0.0.1'], default_keyspace=self.KEYSPACE)
        setup(hosts=['cassandra_api'], default_keyspace=self.KEYSPACE)

    def restart_cassandra(self) -> None:
        self.setup_cassandra()
        self.session.execute('DROP KEYSPACE IF EXISTS ' + self.KEYSPACE)
        self.create_keyspace(self.session)

    def get_session(self) -> Session:
        # cluster = Cluster(['127.0.0.1'], port=9042)
        cluster = Cluster(['cassandra_api'], port=9042)

        session = cluster.connect()
        session.row_factory = dict_factory
        self.create_keyspace(session)
        session.set_keyspace(self.KEYSPACE)
        return session

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
        sample_obj = Sample(
            id=self.LAST_SAMPLE_ID,
            sale=str(sample.get("sale")),
            sales_amount_in_euro=str(sample.get("sales_amount_in_euro")),
            time_delay_for_conversion=str(sample.get("time_delay_for_conversion")),
            click_timestamp=str(sample.get("click_timestamp")),
            nb_clicks_1week=str(sample.get("nb_clicks_1week")),
            product_price=str(sample.get("product_price")),
            product_age_group=str(sample.get("product_age_group")),
            device_type=str(sample.get("device_type")),
            audience_id=str(sample.get("audience_id")),
            product_gender=str(sample.get("product_gender")),
            product_brand=str(sample.get("product_brand")),
            product_category_1=str(sample.get("product_category_1")),
            product_category_2=str(sample.get("product_category_2")),
            product_category_3=str(sample.get("product_category_3")),
            product_category_4=str(sample.get("product_category_4")),
            product_category_5=str(sample.get("product_category_5")),
            product_category_6=str(sample.get("product_category_6")),
            product_category_7=str(sample.get("product_category_7")),
            product_country=str(sample.get("product_country")),
            product_id=str(sample.get("product_id")),
            product_title=str(sample.get("product_title")),
            partner_id=str(sample.get("partner_id")),
            user_id=str(sample.get("user_id")),
            predicted=str(sample.get("predicted")),
            probabilities=json.loads(sample.get("probabilities"))
        )
        self.LAST_SAMPLE_ID += 1
        sample_obj.save()


    def add_some_data(self) -> None:
        # model_history = {
        #     "id": 1,
        #     "name": "new model",
        #     "version": 21,
        #     "timestamp":  1598891820,
        #     # "timestamp":  '2016-04-06 13:06:11.534',
        #     "model": bytes('None', 'utf-8'),
        #     "standard_scaler": bytes('None', 'utf-8'),
        #     "pca_one": bytes('None', 'utf-8'),
        #     "pca_two": bytes('None', 'utf-8')
        # }
        sample = {
            "id": -1,
            "sale": "1",
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
            "user_id": "35"
        }

        # self.insert_model_history(model_history)
        self.insert_sample(sample)

 # todo dodać obsług≥e błedów, jak select nic nie zwraca WSZEDZIE
    def get_last_sample_id(self):
        query = """SELECT MAX(id) FROM sample"""
        query_result = self.session.execute(query)
        result = []
        for sample in query_result:
            result.append(sample)

        return result[0]['system.max(id)']

    def get_all_samples_as_list_of_dicts(self) -> List[dict]:
        return [dict(row) for row in Sample.objects.all()]

    def delete_all_samples(self):
        query = 'TRUNCATE sample'
        self.session.execute(query)

    def get_samples_for_update_model_as_list_of_dicts(self, id):
        if id is None:
            query = """SELECT * FROM sample"""
            query_result = self.session.execute(query)
        else:
            query = """SELECT * FROM sample WHERE id > %s ALLOW FILTERING"""
            query_result = self.session.execute(query, [id])

        result = []
        for sample in query_result:
            result.append(sample)

        print("len")
        print(len(result))

        return result


if __name__ == '__main__':
    db = CassandraClient()
    # db.restart_cassandra()
    # db.add_some_data()
    print(len(db.get_samples_for_update_model_as_list_of_dicts(None)))
    # id = db.get_last_sample_id()
    # print(db.get_samples_for_update_model_as_list_of_dicts(id))
    # print(type(db.get_samples_for_update_model_as_list_of_dicts(id)[0]))
    # print(len(db.get_samples_for_update_model_as_list_of_dicts(id)))