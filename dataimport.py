import pandas as pd
import os
from pathlib import Path
import calendar

def get_input_data(filename):
    data = pd.read_csv(filename, encoding='cp1251')
    data['date'] = pd.to_datetime(data['date'])
    return data

def get_years_df(root):
    df = pd.read_csv(root,encoding='cp1251')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    data_place = {a: 0 for a in range(1950, 2023)}
    start_date = '{year}-01-01'.format(year=list(data_place.keys())[0])
    end_date = '{year}-12-31'.format(year=list(data_place.keys())[-1])
    df.set_index('date', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df_filled = df.reindex(date_range, fill_value=None)
    df_filled.interpolate(limit_direction='both', inplace=True)
    df_filled = df_filled[(df_filled.index.day != 29) | (df_filled.index.month != 2)]
    df_filled = df_filled.sort_index()
    return df_filled

def split_years_to_dict(df):
    columns = df.columns.values.tolist()
    years_dict_by_params = {}
    years = list(set(df.index.year))
    for param in columns:
        param_df = pd.DataFrame(columns=years)
        for i in years:
            tmp = df[param][(df.index.year == i)]
            tmp.reset_index(drop=True, inplace=True)
            param_df[i] = tmp
        years_dict_by_params[param] = param_df

    return years_dict_by_params




