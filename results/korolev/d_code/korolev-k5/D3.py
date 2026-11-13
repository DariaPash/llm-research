from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1. Генерация данных
X, y = make_classification(
    n_samples=5000, 
    n_features=20, 
    n_informative=5, 
    weights=[0.95, 0.05], 
    random_state=42
)

# 2. Разделение на обучающую и тестовую выборки (25% в тест)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Настройка RandomForestClassifier и пространства гиперпараметров
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}

# 4. Подбор гиперпараметров через кросс-валидацию (метрика ROC AUC)
grid_search = GridSearchCV(
    estimator=rf, 
    param_grid=param_grid, 
    scoring='roc_auc', 
    cv=5, 
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 5. Оценка модели на тестовой выборке
best_model = grid_search.best_estimator_
y_probs = best_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_probs)

print(f"ROC AUC on test set: {auc_score}")
