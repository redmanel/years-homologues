import dataimport
import pandas as pd
import matplotlib.pyplot as plt
import calendar
import numpy as np
import time
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from tqdm.autonotebook import tqdm
import datetime
import predict
from pathlib import Path
import os

def get_cluster_kmeans(dataframe,name=''):
    # Подготовка df
    scaler = StandardScaler()
    df = dataframe
    # Кластеризация K-Means
    distortions = []
    silhouette = []
    #print(df)
    if np.any(df.isna().values):
        print('if??')
        df.interpolate(limit_direction='both', inplace=True)
    df_scaled = scaler.fit_transform(df).T
    K = range(1, 10)
    for k in tqdm(K):
        kmeanModel = TimeSeriesKMeans(n_clusters=k, metric="euclidean", n_jobs=6, max_iter=10)
        kmeanModel.fit(df_scaled)
        #Значение для определения n
        distortions.append(kmeanModel.inertia_)
        if k > 1:
            silhouette.append(silhouette_score(df_scaled, kmeanModel.labels_))
    #print('distortions  ', distortions)
    #print('silhouette  ', silhouette)

    if distortions[1] > distortions[0]:
        trend = True
    elif distortions[1] < distortions[0]:
        trend = False
    elif distortions[1] == distortions[0]:
        trend = True
    stop = 0
    n = 1
    b = 1
    while stop != 1:
        if distortions[b] > distortions[b-1]:
            TrendX = True
        elif distortions[b] < distortions[b-1]:
            TrendX = False
        if trend == TrendX:
            stop = 0
            n = n+1
            b= b+1
        if trend != TrendX:
            stop= 1
        if b == 6:
            stop=1
        #print(n)

    # plt.figure(figsize=(10, 4))
    # plt.plot(K, distortions, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Distortion')
    # plt.title('Elbow Method')
    #plt.savefig('distor {name} with n {n}.png'.format(name=name, n=n)) # Сохранение графика в файл для отладки
    #print('distor {name} with n {n}.png'.format(name=name, n=n))

    n_clusters = 6 # Количество кластеров
    ts_kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", n_jobs=3, max_iter=10)
    ts_kmeans.fit(df_scaled)

    df.loc[df.iloc[-1:].index[0]+1] = ts_kmeans.predict(df_scaled)

    return df


def visualize_clusters(years_df, clusters_match):
    clusted_years_list = []
    figs = []
    for ent in clusters_match:
        clusted_years_list.append(ent[0])
    #clusted_years_list.remove(1968) # На время отладки, т.к. берём данные за этот год
    #print(clusted_years_list)
    for param in years_df:
        param_df = years_df[param]
        count = 0
        for year in clusted_years_list:
            plt.plot(param_df[year], label=year, alpha=0.3)

        plt.plot(param_df[1000], label='Целевой год')
        plt.legend()
        # add axis labels and a title
        plt.ylabel(param, fontsize=14)
        plt.xlabel('Номер дня', fontsize=14)
        plt.title(param, fontsize=16)
        gr_dir_path = str(Path(__file__).parent) + '/graphs/'
        if not os.path.exists(gr_dir_path):
            os.mkdir(gr_dir_path)
        fig_name = 'graphs/clusted years for {param}'.format(param=param)
        plt.savefig(fig_name)
        figs.append(fig_name + '.png')
        plt.close()
    return figs





def input_values_to_df(df): #Полученный df должен иметь поле Data в формате datetime
    input_df = df.copy()
    input_df['date'] = pd.to_datetime(input_df['date'])
    year = input_df['date'].iloc[0].year
    # устанавливаем 'Дата' в качестве индекса
    input_df.set_index('date', inplace=True)
    input_df.index = pd.DatetimeIndex(input_df.index)
    date_range = pd.date_range(start='{year}-01-01'.format(year=year), end='{year}-12-31'.format(year=year), freq='D')
    input_df = input_df.reindex(date_range, fill_value=None)

    if calendar.isleap(year):
        day = 29
        month = 2
        ind = input_df.loc[(input_df.index.day == day) & (input_df.index.month == month)].index
        input_df.drop(ind, inplace=True)
    input_df.reset_index(drop= True , inplace= True )
    res = input_df.dropna(how='all')
    #print(res)
    #print(res)
    return res

def cut_df_to_input_range(input, df):
    output = {}
    index_list = list(input.index.values)
    for i in df:
        output[i] = df[i].loc[index_list]
    return output

def insert_input_to_data(input, data):
    # Здесь input должен быть df, который содержит такие же столцы, что и наш набор данных
    colums = input.columns.values.tolist()
    for ent in colums:
        try:
            raw = data[ent]
            raw.insert(loc=len(raw.columns), column=1000, value=input[ent].values)
            data[ent] = raw
        except:
            #print(ent)
            print('Столбы не совпадают')
    return data

def get_all_clusters(dict_years):
    output = {}
    for param in dict_years:
        #print(dict_years[param])
        #print(dict_years[param])
        #print(param)
        output[param] = get_cluster_kmeans(dict_years[param], param)
    return output

# def clear_clusterdf(cluster_df):
#     for param in  cluster_df:
#         param_df = cluster_df[param]
#         columns = param_df.columns.values.tolist()
#         input_col = columns[-1]
#         for col in columns:
#             last_ind = param_df.iloc[-1:].index[0]
#             if param_df[col][last_ind] != param_df[input_col][last_ind]:
#                 del param_df[col]
#         cluster_df[param] = param_df
#     return cluster_df

def count_clusters(cluster_df):
    years_cl_match = {}
    for param in cluster_df:
        param_df = cluster_df[param]
        #target_cluster = param_df[1000].iloc[-1:]
        columns = param_df.columns.values.tolist()
        last_ind = param_df.iloc[-1:].index[0]
        input_col = columns[-1]
        for col in columns:
            if col != 1973:
                if int(param_df[col][last_ind]) == int(param_df[input_col][last_ind]):
                    # print('Param ', param, '  year  ', col, '  cluster_year  ', param_df[col][last_ind], ' cluster_tar ', param_df[input_col][last_ind])
                    try:
                        years_cl_match[col] = years_cl_match[col] + 1
                    except:
                        years_cl_match[col] = 1


    #sorted = heapq.nsmallest(treshold, all_dicts[izm], key=all_dicts[izm].get)

    years_cl_match = {k: v for k, v in years_cl_match.items() if (v >= 4) & (k != 1000) } #Фильтрация значений меньше 3
    years_cl_match = sorted(years_cl_match.items(), key=lambda item: item[1], reverse=True)  # Сортировка по убыванию
    return years_cl_match

def date_re_counter(df, start_date):
    #print(df)
    #print('start date', start_date)
    num_rows = df.shape[0]
    for i in range(0,num_rows):
        #print(start_date + datetime.timedelta(days=i))
        df.rename(index={i: start_date + datetime.timedelta(days=i)}, inplace=True)
    return df

def get_dict_dates(param_df): #Возвращает словрь, который соотносит порядковый номер с датой
    columns = param_df.columns.values.tolist()
    date_dict_range = {}
    for year in columns:
        date_range = pd.date_range(start='{year}-01-01'.format(year=year), end='{year}-12-31'.format(year=year),freq='D')
        daylist = []
        if calendar.isleap(year):
            for i in range(0, 59):
                date_dict_range[date_range[i]] = i + 1
            for i in range(60, 366):
                date_dict_range[date_range[i]] = i
        if not calendar.isleap(year):
            for i in date_range:
                date_dict_range[i] = int(f"{i:%j}")
    return date_dict_range

def rebuild_df_to_range(all_dict, target_df, cluster_match): # Готово. Возвращает словарь параметров, где 1 df - все
                                                            # года гомологи в нужном отрезке времени
    ranged_dict = {}
    clusted_years_list = []
    for ent in cluster_match:
        if ent != 1973: # На время отладки, т.к. 1972 пустой
            clusted_years_list.append(ent[0])
    for param in target_df:
        param_df = all_dict[param]
        columns = param_df.columns.values.tolist()
        target_param = target_df[param]
        # for column in columns:
        #     if column in clusted_years_list:
        #         start_date = '{year}-01-01'.format(year=column)
        #         start_date = pd.to_datetime(start_date)
        #         rebuiled_param = date_re_counter(param_df[column], start_date)
        #         print('Общая база ', rebuiled_param)
        #         print('Df с предсказанием ', target_param)
        start_date = '{year}-01-01'.format(year=1900)
        start_date = pd.to_datetime(start_date)
        rebuiled_param = date_re_counter(param_df, start_date)
        #print(rebuiled_param)
        #print(target_param)
        #print('param ', param_df)
        count_days = 0

        all_years_param_ranged_dict = {}
        #print(target_param.head(1).index[0].day)
        target_start_day = target_param.head(1).index[0].day
        target_start_month = target_param.head(1).index[0].month
        for column in columns:
            if column in clusted_years_list:
                param_ranged_df = pd.DataFrame(columns=[column])
                start_date = '{year}-{month}-{day}'.format(year=column,day=target_start_day, month=target_start_month)
                start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
                end_date = start_date + datetime.timedelta(days=target_param.shape[0])
                date_range = pd.date_range(start=start_date, end=end_date,freq='D')
                dates_dict = get_dict_dates(param_df)
                for i in date_range:
                    if i.day != 29 or i.month != 2:
                        year = i.year
                        int_date = dates_dict[i] - 1
                        # print('date ', i, 'inte ', param_df[year][int_date])
                        param_ranged_df.loc[i] = param_df[year][int_date]
                all_years_param_ranged_dict[column]=param_ranged_df
        #print(param_ranged_df)
        ranged_dict[param] = all_years_param_ranged_dict
    return ranged_dict
    # f"{today:%j}"

