from celery import Celery
import time
import requests

app = Celery('tasks', broker='redis://localhost', backend='redis://localhost')


@app.task
def update_model(data):
    print("Updating model asynchronously...")
    time.sleep(1)
    print("Done updating model.")
    requests.request(method='GET', url='http://127.0.0.1:5000/update_ready', data='new_model_file_name.pkl')
    return None
