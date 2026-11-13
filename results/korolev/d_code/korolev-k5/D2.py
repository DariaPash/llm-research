from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Загрузка датасета
digits = load_digits()
X, y = digits.data, digits.target

# Разделение на обучающую и тестовую выборки (25% в тест, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Создание конвейера (Pipeline): 
# 1. PCA с сохранением 95% дисперсии
# 2. Логистическая регрессия
pipeline = Pipeline([
    ('pca', PCA(n_components=0.95)),
    ('logistic', LogisticRegression(max_iter=10000))
])

# Обучение модели
pipeline.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = pipeline.predict(X_test)

# Вычисление и вывод accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
