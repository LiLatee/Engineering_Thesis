from sklearn.linear_model import SGDClassifier


class ModelInfo:
    id: int = None
    name: str = None
    version: int = None
    date_of_create: float = None
    last_sample_id: str = None
    model: SGDClassifier = None
    LabelEncoders_dict = None
    standard_scaler = None
    df_product_clicks_views = None





