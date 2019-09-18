import zmq
from model_SGDClassifier import ModelSGDClassifier


class ZmqApi:
    def __init__(self):
        self.model = ModelSGDClassifier()
        self.model.load_model()
        self.counter_to_update_model = 0
        self.counter_to_load_model = 0
        self.context = zmq.Context()
        self.fit_socket = self.context.socket(zmq.PUSH).bind("tcp://127.0.0.1:5001")
        self.update_socket = self.context.socket(zmq.PUSH).bind("tcp://127.0.0.1:5002")

    def update_model(self):
        self.update_socket.send_string("update_model")

    def fit_model(self, training_data_json):
        self.fit_socket.send_json(training_data_json)

    def predict(self, sample_json):
        if self.counter_to_load_model >= 200:
            self.model.load_model()
            self.counter_to_load_model = 0
        if self.counter_to_update_model >= 500:
            self.update_model()
            self.counter_to_update_model = 0

        self.predict(sample_json)
        self.counter_to_update_model += 1
        self.counter_to_load_model += 1


