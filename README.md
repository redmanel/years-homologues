# Руководство по запуску приложения


## I. Установка необходимых библиотек
```python
pip install -r requirements.txt
```

## II. Предобученные модели.   
Для ускорения работы предсказания используются предобученные модели.   
Модели **.pkl** находятся по пути: */models/{название станции}/file.pkl*  
Файлы моделей обладают слишком большим объемом для загрузки на github, поэтому существует два варианта работы:  
### Вариант 1.
При выполнении предсказания происходит поиск необходимых файлов. В случае их отсутствия
происходит обучение модели и сохранение полученной модели в соответствующую папку для дальнейшего использования. Если 
нужно отключить сохранение обученных моделей необходимо закомментировать блок в файле **predict.py**  
  
**Predict.py**
```python
# Блок для сохранения результатов обучения в файл .pkl
some code - можно закомментировать для отключения сохранения
# Конец блока
```

### Вариант 2.

В директории **/models/** находится скрипт **download_amderma.py**. Запуск данного скрипта приведёт к скачиванию .7z 
архива с предобученными моделями для станции Amderma. После выполнения скрипта выполнение предсказания ускорится, т.к. 
исчезнет необходимость в обучении модели SARIMAX.

## III. Запуск программы
Для запуска программы необходимо запустить файл **main_sarimax.py**

Файлы **test_input_amderma.csv** и **test_input_Marresalya.csv** приведены как пример *input данных*, которые необходимо
указать на вкладке **Import**. Помимо приведённых, возможно использование иных данных. Input данные должны начинаться 
**01/01/2023** и иметь расширение **.csv**

Директория **/graphs/** содержит графики, которые используются при работе программы. Также данные графики можно 
сохранить отдельно для дальнейшего использования