#!/bin/bash

# Создание данных
python data_creation.py
# Предобработка данных
python data_preprocessing.py
# Подготовка и обучение модели
python model_preparation.py
# Тестирование модели и вывод оценки метрики
python model_testing.py | grep -oE 'Model test accuracy is: [0-9.]+'
