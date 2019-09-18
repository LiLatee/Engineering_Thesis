import time

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class ModelInfo:
    id: int = None
    name: str = None
    version: int = None
    date_of_create: float = None
    last_sample_id: int = None
    model: SGDClassifier = None
    sc: StandardScaler = None
    pca: PCA = None

    # def __init__(self, id: int, name: str, version: int, date_of_create: str, last_sample_id: int,  model: SGDClassifier, standard_scaler: StandardScaler, pca: PCA):
    #     self.id: int = id
    #     self.name: str = name
    #     self.version: int = version
    #     self.date_of_create: float = time.time()
    #     self.last_sample_id: int = last_sample_id
    #     self.model:  SGDClassifier = model
    #     self.sc: StandardScaler = standard_scaler
    #     self.pca: PCA = pca








