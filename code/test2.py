from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Charger les données (à adapter selon ton chemin)
data_path = r"C:\Users\dell\Desktop\ML_Project\DatasetBC\wdbc.data"
df = pd.read_csv(data_path, header=None)
X = df.iloc[:, 2:]  # Supprime les colonnes ID et Diagnosis
y = df.iloc[:, 1].map({'B': 0, 'M': 1})  # B = 0, M = 1

# Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Définir le modèle
knn = KNeighborsClassifier()

# Grille d'hyperparamètres à tester
param_grid = {
    'n_neighbors': [3, 5, 7, 9],       # Tester plusieurs nombres de voisins
    'weights': ['uniform', 'distance'], # Uniform ou pondéré par distance
    'metric': ['euclidean', 'manhattan', 'minkowski'] # Différentes distances
}

# Initialiser GridSearchCV
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Lancer la recherche
grid_search.fit(X_train, y_train)

# Meilleure combinaison d'hyperparamètres
print("\nMeilleurs hyperparamètres trouvés :", grid_search.best_params_)
print("Meilleure précision obtenue :", grid_search.best_score_)

# Évaluer le modèle avec les hyperparamètres optimaux sur le test set
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nPrécision sur le jeu de test : {accuracy:.2f}%")
