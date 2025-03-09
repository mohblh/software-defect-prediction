# Software Defect Prediction

Un projet de **prédiction de défauts logiciels** utilisant des techniques de _Machine Learning_ (apprentissage supervisé et non supervisé) et de **feature selection** pour améliorer la qualité du code.

---

## 1. Présentation Générale du Projet

**Objectif :**  
Développer un système capable de prédire les défauts dans un logiciel, afin d’améliorer la qualité et la fiabilité du code.

**Résumé :**  
- Le projet utilise des techniques de **Machine Learning** (supervisé et non supervisé).  
- Il s’appuie sur différentes méthodes de **feature selection** pour optimiser la précision.  
- L’objectif final est de **repérer les parties du code** susceptibles de contenir des bugs avant la mise en production.

---

## 2. Structure du Repository

```bash
software-defect-prediction/
├── code/
│   ├── test.py
│   ├── test2.py
│   ├── test3.py
│   └── test4.py
├── datasets/
│   └── # (Jeux de données pour les tests)
├── docs/
│   └── # (Documentation, schémas, diagrammes)
├── latex/
│   ├── images/
│   ├── sections/
│   ├── main.tex
│   └── ...
├── CONTRIBUTING.md
├── README.md
└── .gitignore
```

## 3. Étapes Déjà Réalisées

### Mise en place du repository GitHub
- Création d’une structure de base : `code/`, `latex/`, `datasets/`.
- Ajout d’un `.gitignore` pour ignorer les fichiers temporaires.

### Recherche et étude d’algorithmes
- Apprentissage supervisé : **KNN**, **SVM**, **Régression logistique**, **Arbre de décision**, **Forêt aléatoire**.
- Apprentissage non supervisé : **K-Means**, **Clustering hiérarchique**.
- **ACP** (Analyse en Composantes Principales) pour la réduction de dimension.

### Concept de Feature Selection
- Méthodes : **RFE**, **Lasso**, **Chi-square**, **Feature Importance** (Random Forest).
- Amélioration de la précision jusqu’à ~97%.

### Premiers Tests de Performance
- Scripts Python (dans `code/`) pour évaluer différents algorithmes.
- Comparaison des précisions avec/sans feature selection.

### Rédaction du Rapport (LaTeX)
- Structure initiale (`main.tex`), schémas, sections.
- Présentation du pipeline de prétraitement et de formation du modèle.

---

## 4. Approches Algorithmiques

### Algorithmes Supervisés
- **Régression Logistique** : Classification binaire basée sur un modèle statistique.
- **KNN (k-Nearest Neighbors)** : Classification par vote majoritaire des voisins.
- **SVM (Support Vector Machine)** : Recherche d’un hyperplan séparateur optimal.
- **Forêt Aléatoire (Random Forest)** : Ensemble d’arbres de décision robuste au surapprentissage.

### Algorithmes Non Supervisés
- **K-Means** : Regroupement des données en *k* clusters.
- **Clustering Hiérarchique** : Construction d’une hiérarchie de partitions (dendrogramme).

### Sélection de Caractéristiques (Feature Selection)
- **RFE (Recursive Feature Elimination)** : Élimination récursive des features moins pertinentes.
- **Lasso (L1)** : Pénalise les coefficients pour supprimer les variables peu utiles.
- **Importance des Features (Random Forest)** : Évalue la contribution de chaque variable.
- **Chi-Square** : Mesure la dépendance entre la variable cible et les features.

### Prétraitement
- Gestion des valeurs manquantes, normalisation, standardisation.

---

## 5. Étapes Suivantes / Perspectives

### Validation Croisée
- Mettre en place une cross-validation (*k-fold*) pour fiabiliser les résultats.
- Comparer d’autres métriques : **Précision**, **Rappel**, **F1-score**, **AUC**.

### Évaluation et Interprétation
- Analyse des faux positifs/négatifs via matrice de confusion.
- Courbes ROC pour une vision globale de la qualité du modèle.

### Pipeline Complet
- Automatiser le prétraitement, la sélection de caractéristiques et l’entraînement.
- Générer un rapport final (dans `docs/` ou via LaTeX).

### Finalisation du Mémoire
- Rédaction détaillée de la méthodologie, des expérimentations et de la discussion des résultats.
- Présentation des limites et perspectives d’amélioration.

---

## 6. Conclusion

Ce projet vise à **prédire les défauts logiciels** en s’appuyant sur différents algorithmes de Machine Learning et des techniques de sélection de caractéristiques. Les tests préliminaires indiquent des précisions pouvant atteindre **97\%**. Les prochaines étapes incluent une évaluation plus rigoureuse (validation croisée) et la finalisation du rapport en LaTeX.
