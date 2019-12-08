import time

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class ModelInfo:
    id: int = None
    name: str = None
    version: int = None
    date_of_create: float = None
    last_sample_id: str = None
    model: SGDClassifier = None
    sc: StandardScaler = None
    pca: PCA = None





