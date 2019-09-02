from redis_client import set_model_version
from model_SGDClassifier import ModelSGDClassifier

class MLModel:

    def __init__(self) -> None:
        super().__init__()
        self.name = 'Name - TODO'
        self.version = 1
        set_model_version(1)
        self.model = ModelSGDClassifier()

    def fit(self, data):
        print("Model is trained...")
        self.model.create_model_and_save(data)
        print("Done.")
        pass

    def predict(self, sample):
        print("Prediction is being made")
        print("Sample: " + str(sample))
        print('model.version=' + str(self.version))
        return self.model.predict(sample)

    def update_start(self, data_json):
        self.model.update_model()

    def update_ready(self):
        self.update_version()

    def update_version(self):
        self.version += 1
        set_model_version(self.version)

