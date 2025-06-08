from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss

x, y = make_classification(n_samples=100000, n_features=20, n_informative=15, n_redundant=5, n_classes=2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

dt = DecisionTreeClassifier(random_state=42)
knn = KNeighborsClassifier()

param_grid_dt = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [5, 10, 20],
    "min_samples_split": [4, 8, 16],
}

param_grid_knn = {
    'n_neighbors': [3, 4, 5, 7, 9, 10]
}

grid_search_dt = GridSearchCV(dt, param_grid_dt, cv=3, scoring='f1')
grid_search_dt.fit(x_train, y_train)
print("Najlepsze parametry dla drzewa decyzyjnego:", grid_search_dt.best_params_)

grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=3, scoring='f1')
grid_search_knn.fit(x_train, y_train)
print("Najlepsza liczba sąsiadów k-NN:", grid_search_knn.best_params_)

best_dt = grid_search_dt.best_estimator_
best_knn = grid_search_knn.best_estimator_

y_pred_dt = best_dt.predict(x_test)
y_pred_proba_dt = best_dt.predict_proba(x_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
ce_dt = log_loss(y_test, y_pred_proba_dt)
print("\nDrzewo decyzyjne - Accuracy:", round(acc_dt, 4), "| Cross-Entropy:", round(ce_dt, 4))


y_pred_knn = best_knn.predict(x_test)
y_pred_proba_knn = best_knn.predict_proba(x_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
ce_knn = log_loss(y_test, y_pred_proba_knn)
print("k-NN - Accuracy:", round(acc_knn, 4), "| Cross-Entropy:", round(ce_knn, 4))