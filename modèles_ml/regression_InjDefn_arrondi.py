# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 14:01:45 2025

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.optimize import minimize
from functools import partial
from sklearn.metrics import mean_squared_error, r2_score, classification_report,cohen_kappa_score, confusion_matrix, f1_score

# --- 1. CONFIGURATION ---
INPUT_CSV = 'runsafe_ml_ready.csv'
TARGET_NAME = 'InjJoint'
TARGET_COL = TARGET_NAME + '_encoded'

print(f"--- Test de Régression pour {TARGET_NAME} ---")

# --- 2. CHARGEMENT ET SÉPARATION ---
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print("Erreur: Fichier introuvable.")
    exit()

# Identifier les colonnes
# On exclut toutes les cibles (Y) et les identifiants pour ne garder que les features (X)
all_targets = [col for col in df.columns if col.endswith('_encoded')]
identifiers = ['sub_id', 'session_file']
features_cols = [col for col in df.columns if col not in all_targets and col not in identifiers]

X = df[features_cols]
y = df[TARGET_COL] # On ne prend que InjDefn

# Split (Même graine 'random_state' pour comparer avec vos essais précédents)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- ÉTAPE 1 : Optimisation du Modèle (GridSearch) ---
print("Recherche des meilleurs hyperparamètres...")

rf = XGBClassifier(objective='multi:softmax',random_state=42, n_jobs=-1)

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01,0.05,0.1],        # 'None' laisse l'arbre grandir pleinement
    'max_depth': [3, 5, 7],      # Contrôle le lissage
    'subsample': [0.7,0.9],    # Souvent meilleur que 'auto' pour réduire la variance
    'colsample_bytree': [0.7,0.9]
}

# On optimise sur le R2
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Meilleurs paramètres : {grid_search.best_params_}")

# Prédictions continues brutes
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

print(f"Nouveau f1-score (Test) : {f1_score(y_test, y_pred_test, average='macro'):.4f}")

# # --- ÉTAPE 2 : Optimisation des Seuils (Smart Rounding) ---

# class OptimizedRounder(object):
#     def __init__(self):
#         self.coef_ = [0.5, 1.5, 2.5] # Seuils initiaux standards

#     def _kappa_loss(self, coef, X, y):
#         # Fonction de perte : on veut maximiser le Kappa (donc minimiser -Kappa)
#         X_p = np.copy(X)
#         # Découpage selon les seuils a, b, c
#         X_p[X_p < coef[0]] = 0
#         X_p[(X_p >= coef[0]) & (X_p < coef[1])] = 1
#         X_p[(X_p >= coef[1]) & (X_p < coef[2])] = 2
#         X_p[X_p >= coef[2]] = 3
        
#         ll = cohen_kappa_score(y, X_p, weights='quadratic')
#         return -ll

#     def fit(self, X, y):
#         loss_partial = partial(self._kappa_loss, X=X, y=y)
#         initial_coef = [0.5, 1.5, 2.5]
#         # Optimisation des seuils
#         self.coef_ = minimize(loss_partial, initial_coef, method='nelder-mead')['x']

#     def predict(self, X, coef):
#         X_p = np.copy(X)
#         X_p[X_p < coef[0]] = 0
#         X_p[(X_p >= coef[0]) & (X_p < coef[1])] = 1
#         X_p[(X_p >= coef[1]) & (X_p < coef[2])] = 2
#         X_p[X_p >= coef[2]] = 3
#         return X_p.astype(int)

# print("\nOptimisation des seuils d'arrondi...")
# opt = OptimizedRounder()
# # On apprend les seuils sur le TRAIN pour ne pas tricher
# opt.fit(y_pred_train, y_train)
# best_thresholds = opt.coef_

# print(f"Seuils standards : [0.5, 1.5, 2.5]")
# print(f"Seuils optimisés : {best_thresholds}")

# # --- ÉTAPE 3 : Résultats finaux ---
# y_pred_final = opt.predict(y_pred_test, best_thresholds)

print("\n--- Rapport de Classification (Seuils Optimisés) ---")
print(classification_report(y_test, y_pred_test, zero_division=0))

# Affichage Matrice
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap='Greens')
plt.title('Matrice de Confusion Optimiséeé') ##'\nSeuils: {np.round(best_thresholds, 2)}')
plt.ylabel('Valeur Réelle')
plt.xlabel('Prédiction')
plt.show()
# # --- 3. ENTRAÎNEMENT (RANDOM FOREST REGRESSOR) ---
# print("Entraînement du modèle de régression (Random Forest)...")
# # On utilise un Regressor, pas un Classifier
# model = RandomForestRegressor(
#     n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
# )
# model.fit(X_train, y_train)

# # --- 4. PRÉDICTION CONTINUE ---
# print("Calcul des prédictions...")
# y_pred_continuous = model.predict(X_test)

# # Contraindre les valeurs entre 0 et 3 (car une sévérité < 0 ou > 3 n'existe pas)
# y_pred_continuous = np.clip(y_pred_continuous, 0, 3)

# # Afficher quelques exemples
# results_df = pd.DataFrame({
#     'Réel': y_test.values,
#     'Prédit (Continu)': y_pred_continuous.round(2) # Arrondi pour l'affichage
# })
# print("\nExemples de prédictions continues :")
# print(results_df.head(10))

# # --- 5. ÉVALUATION ---

# # A. Évaluation "Régression" (Erreur moyenne)
# mse = mean_squared_error(y_test, y_pred_continuous)
# r2 = r2_score(y_test, y_pred_continuous)
# print(f"\n--- Performance Régression ---")
# print(f"MSE (Erreur Quadratique Moyenne) : {mse:.4f}")
# print(f"R2 Score : {r2:.4f}")

# # B. Évaluation "Classification" (En arrondissant)
# # On arrondit le score continu à l'entier le plus proche pour voir si on trouve la bonne classe
# y_pred_rounded = np.round(y_pred_continuous).astype(int)

# print(f"\n--- Performance après Arrondi (Classification) ---")
# print(classification_report(y_test, y_pred_rounded, zero_division=0))

# # Matrice de confusion
# cm = confusion_matrix(y_test, y_pred_rounded)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title(f'Matrice de Confusion (Régression arrondie) - {TARGET_NAME}')
# plt.xlabel('Prédiction (Arrondie)')
# plt.ylabel('Valeur Réelle')
# plt.show()

# # Histogramme des erreurs
# plt.figure(figsize=(8, 4))
# errors = y_pred_continuous - y_test
# sns.histplot(errors, kde=True)
# plt.title("Distribution des Erreurs (Prédit - Réel)")
# plt.xlabel("Erreur (0 = parfait)")
# plt.show()