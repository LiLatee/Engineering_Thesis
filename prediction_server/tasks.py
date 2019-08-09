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

    db = DatabaseSQLite.DatabaseSQLite()
    df_samples_to_update = db.get_samples_to_update_model()
    df_one_hot_vectors = model.transform_df_into_df_with_one_hot_vectors(df_samples_to_update)

    x = df_one_hot_vectors.iloc[:, 3:].values
    y = df_one_hot_vectors['Sale'].values.ravel()
    print(df_one_hot_vectors)
    x = sc.transform(x)
    x = normalize(x, norm='l2')
    y = np.array([int(i) for i in y])

    model.partial_fit(x, y, classes=np.array([0, 1]))
    model.save_model()

    print("Done updating model.")
    requests.request(method='GET', url='http://engineeringthesis_prediction_server_1:5000/update_ready')
    return None

