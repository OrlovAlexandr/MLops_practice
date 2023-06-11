import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path, output_path, scaler):
    data_df = pd.read_csv(input_path)
    data_scaled = scaler.transform(data_df)
    scaled_df = pd.DataFrame(data_scaled, columns=data_df.columns)
    scaled_df.to_csv(output_path, index=False)


# Загрузка и предобработка обучающих данных
train_files = os.listdir('train')
scaler = StandardScaler()

# Обучение StandardScaler на обучающих данных
for train_file in train_files:
    train_path = os.path.join('train', train_file)
    train_data = pd.read_csv(train_path)
    scaler.partial_fit(train_data)

# Применение StandardScaler к обучающим данным и сохранение
for train_file in train_files:
    train_path = os.path.join('train', train_file)
    scaled_train_path = os.path.join('train', f'scaled_{train_file}')
    preprocess_data(train_path, scaled_train_path, scaler)

# Загрузка и предобработка тестовых данных
test_files = os.listdir('test')

# Применение StandardScaler к тестовым данным и сохранение
for test_file in test_files:
    test_path = os.path.join('test', test_file)
    scaled_test_path = os.path.join('test', f'scaled_{test_file}')
    preprocess_data(test_path, scaled_test_path, scaler)
