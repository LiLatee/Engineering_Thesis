import time
import tasks
from redis_client import set_model_version

class MLModel:

    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
        self.version = 1
        set_model_version(1)

    def fit(self, data):
        print("Model is trained...")
        time.sleep(2)
        print("Done.")
        pass

    def predict(self, sample):
        print("Prediction is being made")
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

