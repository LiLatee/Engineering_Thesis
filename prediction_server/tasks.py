from celery import Celery
import time
import requests

app = Celery('tasks', broker='redis://redis_service', backend='redis://redis_service')


@app.task
def update_model(data):
    print("Updating model asynchronously...")
    time.sleep(1)
    print("Done updating model.")
    requests.request(method='GET', url='http://engineering_thesis_prediction_server_1:5000/update_ready', data='new_model_file_name.pkl')
    return None
