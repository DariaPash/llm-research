import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# Загрузка датасета
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Разделение на обучающую и тестовую выборки (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Определение пространства гиперпараметров для alpha
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

# Создание модели Ridge
ridge = Ridge()

# Настройка кросс-валидации для подбора лучшего alpha
# scoring='neg_mean_absolute_error', так как GridSearchCV максимизирует метрику
grid_search = GridSearchCV(
    estimator=ridge, 
    param_grid=param_grid, 
    scoring='neg_mean_absolute_error', 
    cv=5
)

# Обучение на тренировочной выборке
grid_search.fit(X_train, y_train)

# Выбор лучшей модели
best_model = grid_search.best_estimator_

# Предсказание на тестовой выборке
y_pred = best_model.predict(X_test)

# Вычисление итогового MAE
mae = mean_absolute_error(y_test, y_pred)

print(f"METRIC: {mae}")
