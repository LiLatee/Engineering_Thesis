import time

class MLModel:

    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
        self.version = 1

    def train(self, data):
        print("Model is trained...")
        time.sleep(5)
        print("Done.")
        pass

    def fit(self, sample):
        print("Prediction is being made")
        pass

    def update(self, data):
        print("Model is updated...")
        time.sleep(5)
        print("Done.")
        pass
