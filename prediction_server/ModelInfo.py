from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

class ModelInfo:
    id: int = None
    name: str = None
    version: int = None
    date_of_create: str = None
    last_sample_id: int = None
    binary_model: SGDClassifier = None
    binary_standard_scaler: StandardScaler = None

    def __init__(self, id: int, name: str, version: int, date_of_create: str, last_sample_id: int,  binary_model: SGDClassifier, binary_standard_scaler: StandardScaler):
        self.id: int = id
        self.name: str = name
        self.version: int = version
        self.date_of_create: str = date_of_create
        self.last_sample_id: int = last_sample_id
        self.binary_model:  SGDClassifier = binary_model
        self.binary_standard_scaler: StandardScaler = binary_standard_scaler







