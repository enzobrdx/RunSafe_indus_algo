# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 13:27:42 2025

@author: Lenovo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score, classification_report, f1_score, roc_auc_score
# --- NOUVEAUX IMPORTS POUR PCA ---
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURATION ---
INPUT_CSV = 'runsafe_ml_ready.csv'

print("--- Modèle Optimisé avec PCA (Principal Component Analysis) ---")
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print("Erreur: Fichier introuvable.")
    exit()

# --- 2. DÉFINITION DES CIBLES ET SÉPARATION ---

# Liste ordonnée de VOS cibles spécifiques
ordered_targets = [
    'Injured_Chronic_encoded',
    'InjDefn_encoded', 
    'InjJoint_encoded', 
    'InjJoint2_encoded'
]

# Identifier les colonnes à exclure des features (X)
all_encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
identifiers = ['sub_id', 'session_file']
cols_to_drop = all_encoded_cols + identifiers

# Définition de X (Features) et y (Targets)
X = df.drop(columns=cols_to_drop)
y = df[ordered_targets]

print(f"Features initiales (X) : {X.shape[1]} variables")
print(f"Cibles (y) : {ordered_targets}")

# Split (Même random_state pour comparer)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# --- INTÉGRATION PCA (NOUVEAU BLOC) ---
# =============================================================================
print("\n--- Application de la PCA ---")


# 2. Calcul et Application de la PCA
# n_components=0.95 signifie : "Garde autant de composantes qu'il faut pour expliquer 95% de la variance"
# pca = PCA(n_components=0.95, random_state=42)

# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# print("PCA Terminée :")
# print(f"  -> Nombre de variables avant : {X_train.shape[1]}")
# print(f"  -> Nombre de composantes conservées : {X_train_pca.shape[1]}")
# print(f"  -> Réduction de dimension : -{X_train.shape[1] - X_train_pca.shape[1]} variables")

# # 3. Remplacement des datasets par les versions transformées
# # On remet ça dans des DataFrames pour la propreté (les colonnes s'appellent PC1, PC2...)
# pca_cols = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]
# X_train = pd.DataFrame(X_train_pca, columns=pca_cols, index=X_train.index)
# X_test = pd.DataFrame(X_test_pca, columns=pca_cols, index=X_test.index)
# =============================================================================


# --- 3. DÉFINITION DES MODÈLES OPTIMISÉS (GRIDSEARCH) ---

chain_steps = [
    {
        'target': 'InjDefn_encoded', 
        'type': 'regression',
        'model':  RandomForestRegressor(max_depth= 20, max_features= 1.0, min_samples_leaf= 2, n_estimators= 300, random_state=42, n_jobs=-1)
    },
    {
        'target': 'InjJoint_encoded', 
        'type': 'classification',
        'model': RandomForestClassifier(
            max_depth=20,    
            class_weight='balanced_subsample', 
            max_features=1.0,
            min_samples_leaf=1,
            n_estimators=200,
            random_state=42     
        )
    },
    {
        'target': 'Injured_Chronic_encoded', 
        'type': 'classification',
        'model': RandomForestClassifier(
            max_depth=10,    
            class_weight='balanced', 
            max_features=0.5,
            min_samples_leaf=4,
            n_estimators=100,
            random_state=42     
        )
    },
    {
        'target': 'InjJoint2_encoded', 
        'type': 'classification',
        'model': RandomForestClassifier(
            class_weight= 'balanced', max_depth= 10, max_features= 1.0, min_samples_leaf= 4, n_estimators= 100,random_state=42     
        )
    }
]

# --- 4. BOUCLE D'ENTRAÎNEMENT ET DE PRÉDICTION ---

# Initialisation avec les données PCA
X_train_current = X_train.copy()
X_test_current = X_test.copy()

print("\n--- Début de l'Entraînement sur Composantes Principales ---")

for i, step in enumerate(chain_steps):
    target_col = step['target']
    model_type = step['type']
    model = step['model']
    
    print(f"\nÉtape {i+1}: Traitement de {target_col} ({model_type})")
    
    # A. Sélectionner la cible
    y_train_step = y_train[target_col]
    y_test_step = y_test[target_col]
    
    # B. Entraîner le modèle
    model.fit(X_train_current, y_train_step)
    
    # C. Prédire
    pred_train = model.predict(X_train_current)
    pred_test = model.predict(X_test_current)
    
    # D. Évaluation
    if model_type == 'regression':
        pred_test_rounded = np.clip(np.round(pred_test), 0, 3).astype(int)
        acc = accuracy_score(y_test_step, pred_test_rounded)
        r2 = r2_score(y_test_step, pred_test)
        print(f"    -> R2 Score : {r2:.4f}")
        print(f"    -> Accuracy (arrondie) : {acc:.2%}")
        # print("\n    -> Rapport détaillé (sur valeurs arrondies) :")
        # print(classification_report(y_test_step, pred_test_rounded, zero_division=0))
        
    else:
        acc = accuracy_score(y_test_step, pred_test)
        f1 = f1_score(y_test_step, pred_test, average='weighted', zero_division=0)
        try:
            y_prob = model.predict_proba(X_test_current)
            if y_prob.shape[1] == 2:
                auc_score = roc_auc_score(y_test_step, y_prob[:, 1])
            else:
                auc_score = roc_auc_score(y_test_step, y_prob, multi_class='ovr', average='macro')
                
            print(f"    -> Accuracy          : {acc:.2%}")
            print(f"    -> F1-Score Weighted : {f1:.4f}")
            print(f"    -> AUC-ROC           : {auc_score:.4f}")
            
        except Exception as e:
            print(f"    -> AUC-ROC           : Non calculé ({e})")

    # E. Pas de chaînage (Comme demandé : les prédictions ne sont PAS injectées)
    # Le code reste commenté pour mémoire
    # feature_name = f"pred_{target_col}"
    # X_train_current[feature_name] = pred_train
    # X_test_current[feature_name] = pred_test

print("\n--- Modélisation avec PCA Terminée ---")