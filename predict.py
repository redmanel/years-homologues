from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
import dataimport
import homofind_cluster
import datetime
import warnings
import calendar
from datetime import datetime as dt
import statistics
from pathlib import Path
import os

def date_rebuild(df):

    date_range = pd.date_range(start='1900-01-01', end='1900-12-31', freq='D')
    number_list = [i for i in range(1,366)]
    for param in df:
        for i in number_list:
            df[param].rename(index={i: '{day}'.format(day=date_range[i-1])}, inplace=True)
    return df

def predict(df,n_predict,input_df,homologues='',all_dict='', input_raw_dict='',station_name='', df_val = ''):
    raw_inp = input_raw_dict
    print('n_redict ', n_predict)
    predictions_df = {}
    figs = {}
    for param in df:
        plot_list = []
        param_df = input_df[param]
        str_year = param_df.tail(1).index.year[0]
        start_date = param_df.head(1).index
        start_date = pd.to_datetime(start_date)
        start_date = start_date[0]


        try:
            print('predict ', station_name)
            model_path = str(Path(__file__).parent) + '/models/{station_name}/trained_model_{param}.pkl'.format(param=param,station_name=station_name)
            model_fit = SARIMAXResults.load(model_path)
            up_model = SARIMAX(param_df, order=(1, 1, 1), seasonal_order=(1, 2, 1, 12))
            print('{param}'.format(param=param))
            print('start_filter {time}'.format(time=dt.now()))
            updated_result = up_model.filter(model_fit.params)
        except:
            print('file not found, start model.fit {time}'.format(time=dt.now()))
            model = SARIMAX(param_df, order=(1, 1, 1), seasonal_order=(1, 2, 1, 12))
            updated_result = model.fit(disp=False)

            # Блок для сохранения результатов обучения в файл .pkl
            dir_path = str(Path(__file__).parent) + f'/models/{station_name}/'
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            updated_result.save(dir_path + 'trained_model_{param}.pkl'.format(param=param,station_name=station_name))
            # Конец блока


        print('start forecast {time}'.format(time=dt.now()))
        forecast_future = updated_result.get_forecast(steps=n_predict)
        print('end forecast {time}'.format(time=dt.now()))


        # Создаем новый DataFrame для будущих значений
        future_dates = pd.date_range(start=param_df.iloc[-1:].index[0], periods=n_predict, freq='D') + pd.DateOffset(days=1)
        forecast_df = pd.DataFrame({'Дата': future_dates, param: forecast_future.predicted_mean})
        forecast_df.set_index('Дата', inplace=True)




        # # Для валидации в рамках отладки
        #
        # st = forecast_df.head(1).index[0]
        # st = '{year}-{month}-{day}'.format(year=st.year, month=st.month, day=st.day)
        # en = forecast_df.tail(1).index[0]
        # en = '{year}-{month}-{day}'.format(year=en.year, month=en.month, day=en.day)
        #
        # df_val[param] = df_val[param].query("'{start}' <= index <= '{end}'".format(start=st,end=en))
        # # Рассчитываем MSE и MAE
        # mse = mean_squared_error(df_val[param], forecast_df)
        # mae = mean_absolute_error(df_val[param], forecast_df)
        # r2 = r2_score(df_val[param], forecast_df)
        #
        # #print(f'MSE: {mse}')
        # #print(f'MAE: {mae}')
        # #print(f'R2: {r2}')
        # text_file = open("metric_predict_{n}_days.txt".format(n=n_predict), "a")
        # text_file.write('For {param}: mse = {mse}    mae = {mae}  r2 = {r2} \n'.format(param=param,mse=mse,mae=mae,r2=r2))
        # text_file.close()



        #Присоединяем прогноз к исходному DataFrame
        data_inter = pd.concat([raw_inp[param], forecast_df], ignore_index=False)
        predictions_df[param] = data_inter


        # Визуализация исходных данных и прогноза
        plt.figure(figsize=(12, 6))

        # plot_list.append(plt.plot(first.index, first, label='Исходные данные'))
        # plot_list.append(plt.plot(second.index, second, label='Прогноз'))
        # plot_list.append(plt.plot(df_val[param].index,df_val[param],label='True'))

        ranged_dict = homofind_cluster.rebuild_df_to_range(all_dict,predictions_df,homologues)
        ranged_dict_param = ranged_dict[param]

        for year in ranged_dict_param:
            df_year = ranged_dict_param[year]
            start_homo_year = df_year.head(1).index.year[0]
            for index, row in df_year.iterrows():
                if index.year == start_homo_year:
                    df_year.rename(index={index:index.replace(year=str_year)}, inplace=True)
                if index.year > start_homo_year:
                    diff = index.year - start_homo_year
                    ent = str_year + diff
                    df_year.rename(index={index:index.replace(year=ent)}, inplace=True)

        for year in ranged_dict_param:
            plot_list.append(plt.plot(ranged_dict_param[year].index, ranged_dict_param[year], label=year, alpha=0.3))

        # Попробуем изменить предсказание гомологами

        modified_predict = forecast_df.copy()
        modified_median_predict = forecast_df.copy()
        for index, row in forecast_df.iterrows():
            diff = []
            for year in ranged_dict_param:
                diff.append(forecast_df.loc[index].values[0] - ranged_dict_param[year].loc[index].values[0])
            final_diff = mean(diff)
            final_diff_median = statistics.median(diff)
            modified_predict.loc[index] = forecast_df.loc[index] - final_diff
            modified_median_predict.loc[index]= forecast_df.loc[index] - final_diff_median

        first = raw_inp[param]
        first.loc[modified_predict.head(1).index[0]] = modified_predict.head(1)[param].values[0]

        plot_list.append(plt.plot(first.index, first, label='Исходные данные',color='blue'))
        #plot_list.append(plt.plot(second.index, second, label='Прогноз'))
        #plot_list.append(plt.plot(df_val[param].index,df_val[param],label='True')) # Для валидации настоящие значения
        #plot_list.append(plt.plot(modified_predict.index, modified_predict, label='Modified'))

        plot_list.append(plt.plot(modified_predict.index, modified_predict,
                                  label='Предсказано', color='red'))
        data_inter = pd.concat([raw_inp[param], modified_predict], ignore_index=False)
        predictions_df[param] = data_inter

        # # Метрики для валидации
        #
        # mse_modif = mean_squared_error(df_val[param], modified_predict)
        # mae_modif = mean_absolute_error(df_val[param], modified_predict)
        # r2_modif = r2_score(df_val[param], modified_predict)
        # mse_modif_median = mean_squared_error(df_val[param], modified_median_predict)
        # mae_modif_median = mean_absolute_error(df_val[param], modified_median_predict)
        # r2_modif_median = r2_score(df_val[param], modified_median_predict)
        #
        #
        #
        # #print(f'MSE: {mse}')
        # #print(f'MAE: {mae}')
        # #print(f'R2: {r2}')
        #
        #
        # text_file = open("metric_predict_{n}_days.txt".format(n=n_predict), "a")
        # text_file.write('For {param}: modif_mse = {mmse}    modif_mae = {mmae}  modif_r2 = {mr2} \n'.format(param=param,mmse=mse_modif,mmae=mae_modif,mr2=r2_modif))
        # text_file.close()
        #
        # text_file = open("metric_predict_{n}_days.txt".format(n=n_predict), "a")
        # text_file.write('For {param}: modif_mse_median = {mmse_median}    modif_mae_median = {mmae_median}  modif_r2_median = {mr2_median} \n'.format(param=param,
        #                         mmse_median=mse_modif_median,mmae_median=mae_modif_median,mr2_median=r2_modif_median))
        # text_file.write('\n')
        # text_file.write('\n')
        # text_file.close()


        #Пробуем добавить гомологов

        plt.title('Прогноз {param}'.format(param=param))
        plt.xlabel('Дата')
        plt.ylabel(param)
        plt.legend()
        plt.grid(True)
        gr_dir_path = str(Path(__file__).parent) + '/graphs/'
        if not os.path.exists(gr_dir_path):
            os.mkdir(gr_dir_path)
        fig_name = 'graphs/predict for {param}'.format(n=n_predict,param=param, year=str_year)
        plt.savefig(fig_name)
        figs[param] = fig_name + '.png'
    return predictions_df,figs


def train(df_train):
    text_file = open("pdq Sarimax.txt", "w")
    text_file.close()
    for param in df_train:
        param_df = df_train[param]
        param_df.index = pd.DatetimeIndex(param_df.index).to_period('D')

        model = SARIMAX(param_df, order=(1, 1, 1), seasonal_order=(1, 2, 1, 12))
        print('start model fit time {time}'.format(time=dt.now()))
        model_fit = model.fit()
        print('model fitted time {time}'.format(time=dt.now()))

        # model_fit = model.fit(disp=False)
        model_fit.save('trained_model_{param}.pkl'.format(param=param))

def multi_col_df_to_dict(df):
    columns = df.columns.values.tolist()
    df_dict = {}
    for colm in columns:
      df_dict[colm] = pd.DataFrame(columns=[colm])
    for index, row in df.iterrows():
        for col in columns:
            df_dict[col].loc[index]=row[col]
    return df_dict





