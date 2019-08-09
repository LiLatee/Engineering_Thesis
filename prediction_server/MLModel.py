import time
import tasks
from redis_client import set_model_version
from SGDClassifier import model_SGDClassifier

class MLModel:

    def __init__(self) -> None:
        super().__init__()
        self.name = 'Name - TODO'
        self.version = 1
        set_model_version(1)
        self.model = model_SGDClassifier()

    def fit(self, data):
        print("Model is trained...")
        self.model.create_model_2(data)
        print("Done.")
        pass

    def predict(self, sample):
        print("Prediction is being made")
        print("Sample: " + str(sample))
        print('model.version=' + str(self.version))
        return self.model.predict(sample)

    def update_start(self, data_json):
        tasks.update_model.delay(data_json)

    def update_ready(self, file_name):
        print("Model is replaced with updated one from file " + str(file_name))
        self.update_version()

    def update_version(self):
        self.version += 1
        set_model_version(self.version)

