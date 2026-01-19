# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 21:47:53 2025

@author: Lenovo
"""

import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np

# --- CONFIGURATION ---
INPUT_CSV = 'runsafe_ml_ready.csv'
TARGET_COL = 'InjDefn_encoded' # Votre cible binaire principale
ALPHA = 0.05 # Seuil de significativité (p-value < 0.05)

df = pd.read_csv(INPUT_CSV)

# Identifier les features (exclure cibles et identifiants)
all_targets = [col for col in df.columns if col.endswith('_encoded')]
identifiers = ['sub_id', 'session_file']
feature_cols = [col for col in df.columns if col not in all_targets and col not in identifiers]

# Séparer les groupes
group_injured = df[df[TARGET_COL] > 0]
group_healthy = df[df[TARGET_COL] == 0]

print("--- 1. Test Statistique (Mann-Whitney U) ---")
print(f"Comparaison : Blessés ({len(group_injured)}) vs Sains ({len(group_healthy)})")

significant_features = []

for feature in feature_cols:
    # Récupérer les valeurs
    x_injured = group_injured[feature].dropna()
    x_healthy = group_healthy[feature].dropna()
    
    # Test U
    stat, p_value = mannwhitneyu(x_injured, x_healthy, alternative='two-sided')
    
    # Vérification
    if p_value < ALPHA:
        significant_features.append(feature)
        print(f"[SIGNIFICATIF] {feature:30s} | p-value = {p_value:.5f}")

print(f"\nRésultat : {len(significant_features)} variables retenues sur {len(feature_cols)}.")
print("Liste des variables à utiliser pour l'étape 2 :")
print(significant_features)

# %%

# from sklearn.feature_selection import SequentialFeatureSelector
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import cross_val_score

# # --- CONFIGURATION ---
# # Utilisez la liste obtenue à l'étape 1
# # X_reduced = df[significant_features] 
# # (Pour l'exemple, on suppose que X contient déjà les variables filtrées)
# X = df[significant_features]
# y = df[TARGET_COL]

# # Définir le modèle qui servira à évaluer les subsets
# # (L'article utilisait SVM, mais LogisticRegression est plus rapide pour la sélection)
# estimator = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

# # Configuration de la sélection séquentielle
# sfs = SequentialFeatureSelector(
#     estimator, 
#     n_features_to_select=10, # Laisser l'algo décider du nombre optimal (ou fixer ex: 10)
#     direction='forward',         # Ajouter les variables une par une
#     scoring='r2',                # Optimiser le F1-score (crucial pour le déséquilibre)
#     cv=5,                        # Validation croisée à 5 plis (comme l'article)
#     n_jobs=-1                    # Parallélisation
# )

# print("\n--- 2. Sélection Séquentielle (Forward Selection) ---")
# print("Recherche de la meilleure combinaison de variables...")

# sfs.fit(X, y)

# # Évaluer le modèle avec les variables retenues (ou forcées)
# X_selected = sfs.transform(X)
# scores = cross_val_score(estimator, X_selected, y, cv=5, scoring='r2')

# print(f"\nScore R2 moyen avec les variables sélectionnées : {scores.mean():.4f}")
# print(f"Détail des scores par pli : {scores}")

# # Récupérer les variables choisies
# selected_mask = sfs.get_support()
# final_features = np.array(significant_features)[selected_mask]

# print(f"\nVariables sélectionnées ({len(final_features)}) :")
# print(list(final_features))

# # Créer un nouveau dataset réduit
# X_final = sfs.transform(X)
# print("Forme du dataset final X:", X_final.shape)

# %%

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier # <--- Classifier !
from sklearn.model_selection import StratifiedKFold, cross_val_score

# --- 1. PRÉPARATION DE LA CIBLE BINAIRE ---
# On transforme le problème : 0 = Sain, {1, 2, 3} = Blessé (1)
y_binary = (df[TARGET_COL] > 0).astype(int)

print(f"Distribution de la cible binaire :")
print(y_binary.value_counts(normalize=True))

# --- 2. RÉCUPÉRATION DES FEATURES SIGNIFICATIVES ---
# On reprend votre liste de 23 variables significatives (issues du test Mann-Whitney)
# C'est une excellente base de départ.
significant_features = [
    'speed_r', 'age', 'STRIDE_RATE_median_Avg', 'STRIDE_RATE_std_Avg', 
    'STRIDE_LENGTH_median_Avg', 'STRIDE_LENGTH_std_Avg', 
    'ANKLE_DF_PEAK_ANGLE_median_Avg', 'ANKLE_DF_PEAK_ANGLE_std_Avg', 
    'ANKLE_EVE_PEAK_ANGLE_median_Asymmetry', 'ANKLE_EVE_PEAK_ANGLE_std_Avg', 
    'ANKLE_EVE_PEAK_ANGLE_std_Asymmetry', 'ANKLE_ROT_PEAK_ANGLE_std_Avg', 
    'ANKLE_ROT_PEAK_ANGLE_std_Asymmetry', 'FOOT_PROG_ANGLE_median_Asymmetry', 
    'FOOT_PROG_ANGLE_std_Avg', 'FOOT_PROG_ANGLE_std_Asymmetry', 
    'FOOT_ANG_at_HS_std_Avg', 'VERTICAL_OSCILLATION_median_Asymmetry', 
    'VERTICAL_OSCILLATION_std_Avg', 'PRONATION_ONSET_median_Asymmetry', 
    'PRONATION_OFFSET_median_Avg', 'PRONATION_OFFSET_std_Avg', 
    'PRONATION_OFFSET_std_Asymmetry'
]

X = df[significant_features]

# --- 3. SÉLECTION SÉQUENTIELLE (CLASSIFICATION) ---
print("\n--- Sélection Séquentielle (Classification Binaire) ---")

# On utilise un CLASSIFIER maintenant
# class_weight='balanced' est CRUCIAL car il y a souvent plus de blessés ou de sains
estimator = RandomForestClassifier(
    n_estimators=50, 
    max_depth=5, 
    class_weight='balanced', 
    random_state=42, 
    n_jobs=-1
)

sfs = SequentialFeatureSelector(
    estimator, 
    n_features_to_select=10, # On force 10 variables pour commencer
    direction='forward', 
    scoring='f1',            # On maximise le F1-score (équilibre précision/rappel)
    cv=5,
    n_jobs=-1
)

print("Recherche des 10 meilleures variables pour classifier Blessé vs Sain...")
sfs.fit(X, y_binary)

# Récupération des résultats
selected_mask = sfs.get_support()
final_features = np.array(significant_features)[selected_mask]

print(f"\nVariables sélectionnées pour la Classification ({len(final_features)}) :")
print(list(final_features))

# --- 4. VALIDATION DU SCORE ---
# On vérifie si ce subset donne un bon score
X_selected = sfs.transform(X)
scores = cross_val_score(estimator, X_selected, y_binary, cv=5, scoring='f1')
auc_scores = cross_val_score(estimator, X_selected, y_binary, cv=5, scoring='roc_auc')

print(f"\nPerformance estimée (Cross-Validation) :")
print(f"F1-Score moyen : {scores.mean():.4f}")
print(f"AUC-ROC moyen  : {auc_scores.mean():.4f}")