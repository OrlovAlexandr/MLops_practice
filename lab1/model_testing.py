import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка обученной модели из файла
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Загрузка предобработанных тестовых данных
test_files = [f for f in os.listdir('test') if f.startswith('scaled_')]
X_test = []
y_test = []

for test_file in test_files:
    test_path = os.path.join('test', test_file)
    test_data = pd.read_csv(test_path)
    test_data_x = test_data.index.values % 365 + 1

    X = test_data_x.reshape(-1, 1)
    y = test_data['temperature'].values
    X_test.append(X)
    y_test.append(y)

X_test = np.vstack(X_test)
y_test = np.hstack(y_test)

# Проверка модели на тестовых данных
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error (MSE):', mse)
print('Coefficient of Determination (R^2):', r2)

# Рассчет accuracy
accuracy = model.score(y_test.reshape(-1, 1), y_pred)

print(f"Model test accuracy is: {accuracy:.3f}")
