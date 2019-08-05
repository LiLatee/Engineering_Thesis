import time
import tasks
from redis_client import set_model_version
from model.SGDClassifier import model_SGDClassifier
import pandas as pd
import json

class MLModel:

    def __init__(self) -> None:
        super().__init__()
        self.name = 'SGDClassifier'
        self.SGDClassifier = model_SGDClassifier()
        self.version = 1
        set_model_version(1)

    def fit(self, data):
        print("Model is trained...")
        print(type(data))
        df = pd.DataFrame(data=json.loads(data))
        print('df.shape=' + str(df.shape))

        # print(data)
        self.SGDClassifier.create_model_2(json_data=data)
        print("Done.")
        pass

    def predict(self, sample):
        print("Prediction is being made")
        result = self.SGDClassifier.predict(sample)
        print('result=' + str(result))
        print('model.version=' + str(self.version))
        pass

    def update_start(self, data_json):
        tasks.update_model.delay(data_json)

    def update_ready(self, file_name):
        print("Model is replaced with updated one from file " + str(file_name))
        self.update_version()

    def update_version(self):
        self.version += 1
        set_model_version(self.version)

