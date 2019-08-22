# how to open cqlsh:
# 1. docker exec -it <container_with_cassandra> bash
# 2. cqlsh

from cassandra.cluster import Cluster
from cassandra import util
from typing import NoReturn
from cqlengine import columns
from cqlengine.models import Model
from cqlengine.connection import setup

KEYSPACE = 'keyspace_name'
SAMPLE_TABLE = 'sample'
MODEL_HISTORY_TABLE = 'model_history'
LAST_SAMPLE_ID = 1


class ModelHistory(Model):
    id = columns.Integer(primary_key=True)
    name = columns.Text()
    version = columns.Integer()
    creation_timestamp = columns.Text()
    last_sample_id = columns.Integer()
    model = columns.Bytes()
    standard_scaler = columns.Bytes()


class Sample(Model):
    id = columns.Integer(primary_key=True)
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


def setup_cassandra():
    setup(hosts=['engineeringthesis_cassandra_1'], default_keyspace=KEYSPACE)


def get_session():
    cluster = Cluster(['engineeringthesis_cassandra_1'], port=9042)
    session = cluster.connect()
    create_keyspace(session)
    session.set_keyspace(KEYSPACE)
    return session


def create_keyspace(session):
    session.execute("""
    CREATE KEYSPACE IF NOT EXISTS """ + KEYSPACE + """
    WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '1' }
    """)


def create_tables():
    sql_create_samples_table = """ CREATE TABLE IF NOT EXISTS """ + KEYSPACE + """.""" + SAMPLE_TABLE + """ (
        id INT,
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
        user_id TEXT,
        PRIMARY KEY(id)
        ); """
    sql_create_model_history_table = """ CREATE TABLE IF NOT EXISTS """ + KEYSPACE + """.""" + MODEL_HISTORY_TABLE + """ (
        id INT,
        name TEXT,
        version INT,
        creation_timestamp TIMESTAMP,
        last_sample_id INT,
        model BLOB,
        standard_scaler BLOB,
        PRIMARY KEY(id)
        );
        """

    session = get_session()
    session.execute(sql_create_samples_table)
    session.execute(sql_create_model_history_table)
    return "Tables created properly"


def insert_model_history(model_history: dict) -> NoReturn:
    setup_cassandra()
    model_history_obj = ModelHistory(
        id=int(model_history.get("id")),
        name=str(model_history.get("name")),
        version=int(model_history.get("version")),
        creation_timestamp=str(util.datetime_from_timestamp(model_history.get("timestamp"))),
        last_sample_id=int(model_history.get("last_sample_id")),
        model=model_history.get("model"),
        standard_scaler=model_history.get("standard_scaler"))
    model_history_obj.save()


def insert_sample(sample: dict) -> NoReturn:
    setup_cassandra()
    global LAST_SAMPLE_ID
    sample_obj = Sample(
        id=LAST_SAMPLE_ID,
        sale=str(sample.get("Sale")),
        salesamountineuro=str(sample.get("SalesAmountInEuro")),
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
        user_id=str(sample.get("user_id"))
    )
    LAST_SAMPLE_ID += 1
    sample_obj.save()


def add_some_data():
    model_history = {
        "id": 1,
        "name": "new model",
        "version": 21,
        "timestamp":  1598891820,
        # "timestamp":  '2016-04-06 13:06:11.534',
        "last_sample_id": 583,
        "model": bytes('None', 'utf-8'),
        "standard_scaler": bytes('None', 'utf-8')
    }
    sample = {
        "id": 2,
        "Sale": "1",
        "SalesAmountInEuro": "1",
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

    setup_cassandra()
    session = get_session()
    session.execute('DROP KEYSPACE IF EXISTS ' + KEYSPACE)
    create_keyspace(session)
    create_tables()
    insert_model_history(model_history)
    insert_sample(sample)
    return session.execute("SELECT * FROM system_schema.keyspaces;")


def get_model_history_all():
    setup_cassandra()
    return [dict(row) for row in ModelHistory.objects.all()]


def get_sample_all():
    setup_cassandra()
    return [dict(row) for row in Sample.objects.all()]
