from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Charger les données (à adapter selon ton chemin)
data_path = r"C:\Users\dell\Desktop\ML_Project\DatasetBC\wdbc.data"
df = pd.read_csv(data_path, header=None)
X = df.iloc[:, 2:]  # Supprime les colonnes ID et Diagnosis
y = df.iloc[:, 1].map({'B': 0, 'M': 1})  # B = 0, M = 1

# Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Définir le modèle
knn = KNeighborsClassifier()

# Grille d'hyperparamètres aléatoires à tester
param_dist = {
    'n_neighbors': np.arange(1, 20),  # Nombres de voisins entre 1 et 20
    'weights': ['uniform', 'distance'], # Uniform ou pondéré par distance
    'metric': ['euclidean', 'manhattan', 'minkowski'] # Différentes distances
}

# Initialiser RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=knn, param_distributions=param_dist, 
                                   n_iter=20, cv=5, scoring='accuracy', verbose=1, random_state=42)

# Lancer la recherche
random_search.fit(X_train, y_train)

# Meilleure combinaison d'hyperparamètres
print("\nMeilleurs hyperparamètres trouvés :", random_search.best_params_)
print("Meilleure précision obtenue :", random_search.best_score_)

# Évaluer le modèle avec les hyperparamètres optimaux sur le test set
best_knn = random_search.best_estimator_
y_pred = best_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nPrécision sur le jeu de test : {accuracy:.2f}%")
