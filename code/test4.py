from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
import pandas as pd

# Charger les données
data_path = r"C:\Users\dell\Desktop\ML_Project\DatasetBC\wdbc.data"
df = pd.read_csv(data_path, header=None)
X = df.iloc[:, 2:]  # Supprime les colonnes ID et Diagnosis
y = df.iloc[:, 1].map({'B': 0, 'M': 1})  # B = 0, M = 1

# Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Sélection de caractéristiques
percentage_to_keep = 50  # Garde 50% des caractéristiques
num_features_to_keep = int(X_train.shape[1] * (percentage_to_keep / 100))
selector = SelectKBest(score_func=mutual_info_classif, k=num_features_to_keep)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Récupérer les noms des caractéristiques sélectionnées
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print(f"Caractéristiques sélectionnées (sur {percentage_to_keep}%): {list(selected_features)}")

# Définir le modèle
knn = KNeighborsClassifier()

# Grid Search pour optimiser les hyperparamètres
param_grid = {
    'n_neighbors': [3, 5, 7, 9],       # Tester plusieurs nombres de voisins
    'weights': ['uniform', 'distance'], # Uniform ou pondéré par distance
    'metric': ['euclidean', 'manhattan', 'minkowski'] # Différentes distances
}

grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_selected, y_train)

# Meilleure combinaison d'hyperparamètres
print("\nMeilleurs hyperparamètres trouvés :", grid_search.best_params_)
print("Meilleure précision obtenue pendant la validation croisée :", grid_search.best_score_)

# Évaluer le modèle avec les hyperparamètres optimaux sur le jeu de test
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nPrécision sur le jeu de test : {accuracy:.2f}%")
