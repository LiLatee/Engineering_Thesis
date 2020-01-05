import pandas as pd
import threading

not_sorted_data_file_name = '../data/CriteoSearchData.csv'
train_model_samples_number = 50000
headers = ['sale', 'sales_amount_in_euro', 'time_delay_for_conversion', 'click_timestamp', 'nb_clicks_1week',
           'product_price', 'product_age_group', 'device_type', 'audience_id', 'product_gender',
           'product_brand', 'product_category_1', 'product_category_2', 'product_category_3',
           'product_category_4', 'product_category_5', 'product_category_6', 'product_category_7',
           'product_country', 'product_id', 'product_title', 'partner_id', 'user_id']
options = {
    'filepath_or_buffer': 'data/CriteoSearchData-sorted-no-duplicates-LabelEncoded-with-product-id.csv',
    "chunksize": 3,
    "nrows": 1000000,
    "skiprows": train_model_samples_number,
    "sep": ',',
    "usecols": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    "header": 0,
    "names": headers,
}


class ThreadsafeIterator:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(generator_function):
    def wrapper_function(*args, **kwargs):
        return ThreadsafeIterator(generator_function(*args, **kwargs))

    return wrapper_function


@threadsafe_generator
def data_generator():
    for chunk in pd.read_csv(**options):
        for index, row in chunk.iterrows():
            yield row


def get_train_data():
    return pd.read_csv(
        filepath_or_buffer=options.get('filepath_or_buffer'),
        sep=options.get('sep'),
        nrows=options.get('skiprows'),
        header=options.get('header'),
        usecols=options.get('usecols'))
