from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Загрузка датасета
data = load_wine()
X, y = data.data, data.target

# 2. Разделение на обучающую (75%) и тестовую (25%) выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. Подбор оптимального k
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
best_k = 1
best_score = 0

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    # 5-блочная кросс-валидация на обучающей выборке
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    mean_score = scores.mean()

    if mean_score > best_score:
        best_score = mean_score
        best_k = k

# 4. Обучение модели с лучшим k на всей обучающей выборке
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)

# 5. Вычисление accuracy на тестовой выборке
y_pred = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)

print(f"METRIC: {final_accuracy}")
