from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

x, y = make_classification(n_samples=100000, n_features=20, n_informative=15, n_redundant=5, n_classes=2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

dt = DecisionTreeClassifier(random_state=42)
knn = KNeighborsClassifier()

param_grid_dt = {
    'max_depth': [2, 3, 5, 10, 12, None]
}

param_grid_knn = {
    'n_neighbors': [3, 4, 5, 7, 9, 10, 12]
}

grid_search_dt = GridSearchCV(dt, param_grid_dt, cv=3, scoring='f1')
grid_search_dt.fit(x_train, y_train)
print("Najlepsza głębokość dla drzewa decyzyjnego:", grid_search_dt.best_params_)

grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=3, scoring='f1')
grid_search_knn.fit(x_train, y_train)
print("Najlepsza liczba sąsiadów k-NN:", grid_search_knn.best_params_)