import time
import requests
from redis import Redis
from rq import Queue

redis_conn = Redis(host='redis_service', port=6379, db=0)
q = Queue('queue_update_model', connection=redis_conn)

def run_update_model(data):
    q.enqueue(update_model, data)


def update_model(data):
    print("Updating model asynchronously...")
    time.sleep(1)
    print("Done updating model.")
    requests.request(method='GET', url='http://engineeringthesis_prediction_server_1:5000/update_ready', data='new_model_file_name.pkl')
    return None

