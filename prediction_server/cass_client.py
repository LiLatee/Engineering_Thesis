# how to open cqlsh:
# 1. docker exec -it <container_with_cassandra> bash
# 2. cqlsh


from cassandra.cluster import Cluster
from cassandra.query import dict_factory
import json
import time
import logging
from typing import List, Dict, NoReturn, Union, Any, Optional, Tuple

KEYSPACE = 'keyspace_name'
SAMPLES_TABLE = 'samples'
MODEL_HISTORY_TABLE = 'model_history'

def get_session():
    cluster = Cluster(['engineeringthesis_cassandra_1'], port=9042)
    # cluster = Cluster(['0.0.0.0'], port=9042)
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
    sql_create_samples_table = """ CREATE TABLE IF NOT EXISTS """ + KEYSPACE + """.""" + SAMPLES_TABLE + """ (
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
    timestamp TIMESTAMP,
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
    session = get_session()
    stmt = session.prepare(""" INSERT INTO """ + KEYSPACE + """.""" + MODEL_HISTORY_TABLE + """("id", "name", "version", "timestamp", "last_sample_id", "model", "standard_scaler")
        VALUES (?, ?, ?, ?, ?, ?, ?)""")
    session.execute(stmt, [int(model_history.get("id")),
        str(model_history.get("name")),
        int(model_history.get("version")),
        model_history.get("timestamp"),
        int(model_history.get("last_sample_id")),
        model_history.get("model"),
        model_history.get("standard_scaler")])


def insert_sample(sample: dict) -> NoReturn:
    session = get_session()
    stmt = session.prepare(""" INSERT INTO """ + KEYSPACE + """.""" + SAMPLES_TABLE + """(
            "Sale",
            "SalesAmountInEuro",
            "time_delay_for_conversion",
            "click_timestamp",
            "nb_clicks_1week",
            "product_price",
            "product_age_group",
            "device_type",
            "audience_id",
            "product_gender",
            "product_brand",
            "product_category_1",
            "product_category_2",
            "product_category_3",
            "product_category_4",
            "product_category_5",
            "product_category_6",
            "product_category_7",
            "product_country",
            "product_id",
            "product_title",
            "partner_id",
            "user_id" )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""")

    # sample_array = sample.values()
    session.execute(stmt,
        [sample.get("Sale"),
        sample.get("SalesAmountInEuro"),
        sample.get("time_delay_for_conversion"),
        sample.get("click_timestamp"),
        sample.get("nb_clicks_1week"),
        sample.get("product_price"),
        sample.get("product_age_group"),
        sample.get("device_type"),
        sample.get("audience_id"),
        sample.get("product_gender"),
        sample.get("product_brand"),
        sample.get("product_category_1"),
        sample.get("product_category_2"),
        sample.get("product_category_3"),
        sample.get("product_category_4"),
        sample.get("product_category_5"),
        sample.get("product_category_6"),
        sample.get("product_category_7"),
        sample.get("product_country"),
        sample.get("product_id"),
        sample.get("product_title"),
        sample.get("partner_id"),
        sample.get("user_id")])

def get_data():
    model_history = {
        "id": 1,
        "name": "new model",
        "version": 2,
        "timestamp": time.time(),
        "last_sample_id": 583,
        "model": bytes('None', 'utf-8'),
        "standard_scaler": bytes('None', 'utf-8')
    }
    sample = {
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

    session = get_session()
    create_keyspace(session)
    create_tables()
    insert_sample(sample)
    # insert_model_history(model_history)
    return session.execute("SELECT * FROM system_schema.keyspaces;")
    # return session.execute("SELECT * FROM " + KEYSPACE + "." + MODEL_HISTORY_TABLE + ";")
