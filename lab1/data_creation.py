import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Генерация синтетических данных
def generate_temperature_data(days=365, num_years=10, noise_std=2):
    temperature_data = []
    for _ in range(num_years):
        base_temp = 20 + np.random.rand() * 10  # Случайная базовая температура
        daily_temp = np.sin(np.linspace(0, 2 * np.pi, days)) * 10 + base_temp
        noise = np.random.normal(0, noise_std, days)
        temperature_data.extend(daily_temp + noise)
    return np.array(temperature_data)


# Создание папок
if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('test'):
    os.makedirs('test')

# Генерация и сохранение данных
num_datasets = 3
for i in range(num_datasets):
    data = generate_temperature_data()
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)

    train_df = pd.DataFrame(train_data, columns=['temperature'])
    train_path = os.path.join('train', f'train_data_{i}.csv')
    train_df.to_csv(train_path, index=False)

    test_df = pd.DataFrame(test_data, columns=['temperature'])
    test_path = os.path.join('test', f'test_data_{i}.csv')
    test_df.to_csv(test_path, index=False)
