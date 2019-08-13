import time
import requests
from redis import Redis
from rq import Queue
import DatabaseSQLite
import numpy as np
from sklearn.preprocessing import normalize

redis_conn = Redis(host='redis_service', port=6379, db=0)
q = Queue('queue_update_model', connection=redis_conn)

def run_update_model(model, sc):
    q.enqueue(update_model, model, sc)


def update_model(model, sc):
    print("Updating model asynchronously...")
    time.sleep(1)
    print("Done updating model.")
    requests.request(method='GET', url='http://engineeringthesis_prediction_server_1:5000/update_ready')
    return None
