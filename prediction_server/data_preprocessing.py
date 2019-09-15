import numpy as np
import json
import os

from typing import List, Dict, Union, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from imblearn.over_sampling import ADASYN
from sklearn.decomposition import PCA


JSONType = Union[str, bytes, bytearray]
RowAsDictType = Dict[str, Union[str, float, int]]


def create_train_and_test_sets(training_data_json: JSONType) -> List[np.ndarray]:
    # print("create_train_and_test_sets")
    data_as_list_of_dicts = json.loads(training_data_json)

    x, y = split_data_to_x_and_y(data_as_list_of_dicts)

    # x = df_one_hot_vectors.iloc[:, 3:].values
    # y = df_one_hot_vectors.iloc[:, :1].values.ravel()  # tutaj powinny być chyba 3 kolumny

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

    print('Liczba etykiet w zbiorze y:', np.bincount(y))
    print('Liczba etykiet w zbiorze y_train:', np.bincount(y_train))
    print('Liczba etykiet w zbiorze y_test:', np.bincount(y_test))

    pca = PCA(n_components=500, random_state=1)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    adasyn = ADASYN(random_state=1)
    x_train, y_train = adasyn.fit_resample(x_train, y_train)

    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_train_std = normalize(x_train_std, norm='l2')
    x_test_std = sc.transform(x_test)
    x_test_std = normalize(x_test_std, norm='l2')

    return [pca, sc, x_test_std, x_train_std, y_test, y_train]

def split_data_to_x_and_y(data_as_list_of_dicts: List[RowAsDictType]) -> Tuple[np.ndarray, np.ndarray]:
    list_of_dicts_of_samples = transform_list_of_dicts_to_list_of_one_hot_vectors_dicts(data_as_list_of_dicts)
    x = [list(s.values())[3:] for s in list_of_dicts_of_samples]
    y = [list(s.values())[:1][0] for s in list_of_dicts_of_samples]
    x = np.array(x)
    y = np.array(y)
    return x, y

def transform_list_of_dicts_to_list_of_one_hot_vectors_dicts(list_of_dicts: List[RowAsDictType]) -> List[RowAsDictType]:
    # print("transform_df_into_df_with_one_hot_vectors")
    samples = []
    for row_as_dict in list_of_dicts:
        samples.append(transform_dict_row_in_one_hot_vectors_dict(row_as_dict))

    return samples

def transform_dict_row_in_one_hot_vectors_dict(row_as_dict: RowAsDictType) -> RowAsDictType:
    # print('transform_json_row_in_one_hot_vectors_dict')

    transformed_row_as_dict = create_dict_as_transformed_row_and_set_no_one_hot_vectors_columns(row_as_dict)
    transformed_row_as_dict = set_values_to_one_hot_vectors_columns(row_as_dict, transformed_row_as_dict)

    return transformed_row_as_dict

def create_dict_as_transformed_row_and_set_no_one_hot_vectors_columns(old_dict: RowAsDictType) -> RowAsDictType:
    # print('create_dict_as_transformed_row_and_set_no_one_hot_vectors_columns')
    required_column_names_list = read_required_column_names()
    new_dict: RowAsDictType = dict.fromkeys(required_column_names_list, 0)
    new_dict['sale'] = int(old_dict['sale'])

    new_dict['sales_amount_in_euro'] = old_dict['sales_amount_in_euro']
    new_dict['time_delay_for_conversion'] = int(old_dict[
        'time_delay_for_conversion'])
    new_dict['click_timestamp'] = int(old_dict['click_timestamp'])
    new_dict['nb_clicks_1week'] = int(old_dict['nb_clicks_1week'])

    return new_dict

def set_values_to_one_hot_vectors_columns(old_dict: RowAsDictType, new_dict: RowAsDictType) -> RowAsDictType:
    # print('set_values_to_one_hot_vectors_columns')
    required_column_names_list = read_required_column_names()
    for column_number, (column_name, cell_value) in enumerate(old_dict.items()):
        if column_number > 2: # skip y rows (sale,  sales_amount_in_euro,time_delay_for_conversion)
            transformed_column_name = column_name + '_' + str(cell_value)
            if transformed_column_name in required_column_names_list:
                new_dict[transformed_column_name] = 1

    return new_dict

def read_required_column_names() -> List[str]:
    curdir = os.getcwd()
    required_column_name_file = open(curdir + '/required_column_names_list.txt', 'r')
    required_column_names_list = required_column_name_file.read().splitlines()
    return required_column_names_list