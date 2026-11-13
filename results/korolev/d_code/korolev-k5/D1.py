from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Разделение на обучающую и тестовую выборки (25% в тест)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. Масштабирование признаков 
# Важно для SVM, так как он чувствителен к масштабу данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Настройка SVM с RBF-ядром и подбор параметра C через кросс-валидацию
param_grid = {
    'C': [0.1, 1, 10, 100, 1000]
}

grid_search = GridSearchCV(
    estimator=SVC(kernel='rbf'), 
    param_grid=param_grid, 
    cv=5, 
    scoring='accuracy'
)
grid_search.fit(X_train_scaled, y_train)

# 5. Оценка на отложенной тестовой выборке
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print(f"Best C: {grid_search.best_params_['C']}")
print(f"Accuracy on test set: {accuracy}")
