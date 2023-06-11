import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Загрузка и обучение модели на предобработанных данных
train_files = [f for f in os.listdir('train') if f.startswith('scaled_')]
X_train = []
y_train = []

for train_file in train_files:
    train_path = os.path.join('train', train_file)
    train_data = pd.read_csv(train_path)
    train_data_x = train_data.index.values % 365 + 1

    X = train_data_x.reshape(-1, 1)
    y = train_data['temperature'].values
    X_train.append(X)
    y_train.append(y)
X_train = np.vstack(X_train)
y_train = np.hstack(y_train)

model = LinearRegression()
model.fit(X_train, y_train)

# Сохранение обученной модели в файл
with open('trained_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
