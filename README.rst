Projet Prédiction du Prix des Voitures BMW
==========================================

Ce projet de machine learning vise à prédire le prix des voitures BMW ainsi que leur catégorie de prix (Cheap, Medium, Expensive) à partir de données de ventes internationales couvrant la période 2010-2024.

Structure du projet
-------------------

- **BMW sales data (2010-2024) (1) (1).csv** : Jeu de données principal, contenant les informations sur les ventes BMW (modèle, année, région, couleur, carburant, transmission, taille moteur, kilométrage, prix, volume de ventes, classification).
- **Test/car_price_prediction.py** : Script principal pour le prétraitement, l’entraînement des modèles, l’évaluation et la visualisation.
- **feature_importance_base_features.png** : Graphique généré automatiquement illustrant l’importance des variables dans la prédiction.
- **Exploration.ipynb** : Notebook Jupyter pour l’exploration, la visualisation et des tests complémentaires.
- **README.rst** : Ce fichier de documentation.

Données utilisées
-----------------

Le fichier [BMW sales data (2010-2024) (1) (1).csv](BMW%20sales%20data%20%282010-2024%29%20%281%29%20%281%29.csv) contient les colonnes suivantes :
- Model, Year, Region, Color, Fuel_Type, Transmission, Engine_Size_L, Mileage_KM, Price_USD, Sales_Volume, Sales_Classification

Le script ajoute des variables dérivées comme l’âge du véhicule, le kilométrage par an et une estimation de la puissance moteur.

Modèles de Machine Learning utilisés
------------------------------------

Le script utilise et compare plusieurs modèles pour la régression (prédiction du prix) et la classification (catégorie de prix) :

**Pour la régression :**
- Linear Regression (Régression Linéaire)
- Decision Tree Regressor (Arbre de Décision)
- K-Nearest Neighbors Regressor (KNN)
- Gradient Boosting Regressor
- Multi-layer Perceptron Regressor (MLP)
- Support Vector Regressor (SVR)
- XGBoost Regressor

**Pour la classification :**
- Logistic Regression (Régression Logistique)
- Gaussian Naive Bayes
- Decision Tree Classifier (Arbre de Décision)
- Gradient Boosting Classifier
- Multi-layer Perceptron Classifier (MLP)
- Support Vector Classifier (SVM)
- XGBoost Classifier

Fonctionnalités principales
---------------------------

- **Prétraitement** : Nettoyage des données, encodage des variables catégorielles, normalisation des variables numériques, création de nouvelles features.
- **Modélisation** : Entraînement, comparaison et évaluation des modèles listés ci-dessus.
- **Évaluation** : Calcul des métriques (RMSE, MAE, R2 pour la régression ; Accuracy, Precision, Recall, F1 pour la classification).
- **Analyse d’importance des variables** : Visualisation des features les plus influentes sur le prix.
- **Visualisation** : Graphiques générés avec Matplotlib et Seaborn.
- **Exploration avancée** : Notebook Jupyter pour analyses complémentaires (clustering, visualisation interactive, etc.).

Utilisation
-----------

1. Installer les dépendances nécessaires :
   ```
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
   ```
2. Lancer le script principal :
   ```
   python Test/car_price_prediction.py
   ```
3. Les résultats et graphiques seront affichés dans la console et sauvegardés dans le dossier du projet.

Exemple de résultats
--------------------

- Prédiction du prix d’une voiture BMW selon ses caractéristiques.
- Classification automatique en catégories de prix.
- Visualisation de l’importance des variables (voir le fichier feature_importance_base_features.png).

Auteurs et contact
------------------

Projet réalisé par Ghaouti CHERCHAB, Mohamed LAIDAOUI & Haythem TOUBALE.  

N’hésitez pas à proposer des améliorations ou à poser des questions !