import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Chemins vers les fichiers (à adapter selon ton système)
data_path = r"C:\Users\dell\Desktop\ML_Project\DatasetBC\wdbc.data"
names_path = r"C:\Users\dell\Desktop\ML_Project\DatasetBC\wdbc.names"

# 2. Charger les noms de colonnes depuis le fichier .names
# Ouvrir et lire les colonnes définies dans wdbc.names
with open(names_path, 'r') as f:
    lines = f.readlines()
    # Exemple : définir manuellement les noms des colonnes d'après le fichier .names
    column_names = [
        "ID", "Diagnosis",  # Colonne d'identification et étiquette
        "Radius_mean", "Texture_mean", "Perimeter_mean", "Area_mean", "Smoothness_mean", 
        "Compactness_mean", "Concavity_mean", "Concave_points_mean", "Symmetry_mean", 
        "Fractal_dimension_mean", "Radius_se", "Texture_se", "Perimeter_se", "Area_se", 
        "Smoothness_se", "Compactness_se", "Concavity_se", "Concave_points_se", "Symmetry_se", 
        "Fractal_dimension_se", "Radius_worst", "Texture_worst", "Perimeter_worst", "Area_worst", 
        "Smoothness_worst", "Compactness_worst", "Concavity_worst", "Concave_points_worst", 
        "Symmetry_worst", "Fractal_dimension_worst"
    ]

# 3. Charger le fichier .data avec les colonnes
df = pd.read_csv(data_path, header=None, names=column_names)

# 4. Séparer les caractéristiques et les étiquettes
X = df.drop(columns=["ID", "Diagnosis"])  # Supprimer ID et étiquette
y = df["Diagnosis"]  # Utiliser la colonne "Diagnosis" comme étiquette

# 5. Encodage des étiquettes : B (bénin) = 0, M (malin) = 1
y = y.map({'B': 0, 'M': 1})

# 6. Sélection de caractéristiques
percentage_to_keep = 70  # Conserver 50% des caractéristiques
num_features_to_keep = int(X.shape[1] * (percentage_to_keep / 100))

# Sélectionner les meilleures caractéristiques avec l'information mutuelle
selector = SelectKBest(score_func=mutual_info_classif, k=num_features_to_keep)
X_reduced = selector.fit_transform(X, y)

# Afficher les caractéristiques sélectionnées
selected_features = np.array(X.columns)[selector.get_support()]
print(f"\nCaractéristiques sélectionnées (sur {percentage_to_keep}%): {selected_features}")

# 7. Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

# 8. Créer un modèle KNN
knn = KNeighborsClassifier(n_neighbors=3)  # k=3
knn.fit(X_train, y_train)

# 9. Évaluer le modèle
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100  # Précision en pourcentage
print(f"\nPrécision du modèle KNN après sélection des caractéristiques : {accuracy:.2f}%")