import homofind_cluster
import predict
import dataimport
import pandas as pd
import warnings
from pathlib import Path
from datetime import datetime as dt

def dict_to_df(dict):
    result = pd.DataFrame(columns=dict.keys())
    for param in dict:
        result[param] = dict[param]
    return result

def homofind(data_root,input_root):
    df = dataimport.get_years_df(data_root)
    years_dict = dataimport.split_years_to_dict(df)
    # Input data
    input = dataimport.get_input_data(input_root)
    input_clust = input.copy()
    input_clust = homofind_cluster.input_values_to_df(input_clust)
    years_dict = homofind_cluster.cut_df_to_input_range(input_clust, years_dict)
    years_dict = homofind_cluster.insert_input_to_data(input_clust, years_dict)
    years_clusters = homofind_cluster.get_all_clusters(years_dict)
    clusters_match_info = homofind_cluster.count_clusters(years_clusters)
    return clusters_match_info, years_dict


def get_predicts(data_root,input_root, n_days, clusters_match_info,station_name):
    print('Pred func start {time}'.format(time=dt.now()))
    warnings.filterwarnings('ignore')
    df = dataimport.get_years_df(data_root)
    years_dict = dataimport.split_years_to_dict(df)
    const_years_dict = years_dict.copy()
    print('years data ready')
    # Input data
    input = dataimport.get_input_data(input_root)
    input_pred = input.copy()
    input_raw_dict = input.copy()
    input_raw_dict.set_index('date', inplace=True)
    input_raw_dict = predict.multi_col_df_to_dict(input_raw_dict)
    input_pred.set_index('date', inplace=True)
    input_pred.index = pd.DatetimeIndex(input_pred.index)
    df_pred = df.query("'1990-01-01' <= index <= '2022-12-31'")
    input_expanded = pd.concat([df_pred, input_pred], ignore_index=False)
    input_expanded = predict.multi_col_df_to_dict(input_expanded)
    print('input data ready')
    # END Input

    # # Валидационный набор 2023
    # input_root = r'H:\Univer\stud\Магистратура\Лед Ситроникс\Pp_gomolog\homologues\alexey\validation_full_2023.csv'
    # df_val = dataimport.get_input_data(input_root)
    # df_val.set_index('date', inplace=True)
    # df_val.index = pd.DatetimeIndex(df_val.index)
    # dict_val = predict.multi_col_df_to_dict(df_val)
    # # print('dd ',dict_val)
    # # END Валидационный

    years_dict = predict.date_rebuild(years_dict)
    print('Prediction start {time}'.format(time=dt.now()))
    prediсtions, figs = predict.predict(years_dict, n_days, input_expanded, clusters_match_info,
                                                  const_years_dict, input_raw_dict, station_name)
    return prediсtions, figs


def train_test(root_data): # Обучение модели на историчеких данных
    df = dataimport.get_years_df(root_data)
    df_ranged = df.copy()
    train_dict = predict.multi_col_df_to_dict(df_ranged)
    print(train_dict)
    predict.train(train_dict)

