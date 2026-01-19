# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 14:30:08 2026

@author: Lenovo
"""
"""
Modèle Optimisé avec Analyse SHAP intégrée
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap  # <--- IMPORT SHAP NÉCESSAIRE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, classification_report, f1_score, roc_auc_score

# --- 1. CONFIGURATION ---
INPUT_CSV = 'runsafe_ml_ready.csv'

print("--- Chaîne Hybride Personnalisée (4 Variables) + SHAP ---")
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print("Erreur: Fichier introuvable.")
    exit()

# --- 2. DÉFINITION DES CIBLES ET SÉPARATION ---

ordered_targets = [
    'Injured_Chronic_encoded',
    'InjDefn_encoded', 
    'InjJoint_encoded', 
    'InjJoint2_encoded'
]

all_encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
identifiers = ['sub_id', 'session_file']
cols_to_drop = all_encoded_cols + identifiers

X = df.drop(columns=cols_to_drop)
y = df[ordered_targets]

print(f"Features (X) : {X.shape[1]} variables")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. DÉFINITION DE LA CHAÎNE ---

chain_steps = [
    {
        'target': 'InjDefn_encoded', 
        'type': 'regression',
        'model':  RandomForestRegressor(max_depth=20, max_features=1.0, min_samples_leaf=2, n_estimators=300, random_state=42, n_jobs=-1)
    },
    {
        'target': 'InjJoint_encoded', 
        'type': 'classification',
        'model': RandomForestClassifier(max_depth=20, class_weight='balanced_subsample', max_features=1.0, min_samples_leaf=1, n_estimators=200, random_state=42)
    },
    {
        'target': 'Injured_Chronic_encoded', 
        'type': 'classification',
        'model': RandomForestClassifier(max_depth=10, class_weight='balanced', max_features=0.5, min_samples_leaf=4, n_estimators=100, random_state=42)
    },
    {
        'target': 'InjJoint2_encoded', 
        'type': 'classification',
        'model': RandomForestClassifier(class_weight='balanced', max_depth=10, max_features=1.0, min_samples_leaf=4, n_estimators=100, random_state=42)
    }
]

# --- 4. BOUCLE D'ENTRAÎNEMENT ---

X_train_current = X_train.copy()
X_test_current = X_test.copy()

print("\n--- Début de la Chaîne ---")

for i, step in enumerate(chain_steps):
    target_col = step['target']
    model_type = step['type']
    model = step['model']
    
    print(f"\nÉtape {i+1}: Traitement de {target_col} ({model_type})")
    
    # A. Sélection Cible
    y_train_step = y_train[target_col]
    y_test_step = y_test[target_col]
    
    # B. Entraînement
    model.fit(X_train_current, y_train_step)
    
    # --- ANALYSE SHAP ---
    print(f"    -> Calcul des valeurs SHAP pour {target_col}...")
    try:
        # On utilise TreeExplainer pour les Random Forest (très rapide et optimisé)
        explainer = shap.TreeExplainer(model)
        
        # On calcule sur le jeu de TEST pour voir la généralisation
        # Si c'est trop lent, vous pouvez prendre un échantillon : X_test_current.sample(100)
        shap_values = explainer.shap_values(X_test_current)
        
        plt.figure(figsize=(10, 6))
        
        # Cas 1 : Régression (shap_values est un tableau unique)
        if model_type == 'regression':
            plt.title(f"SHAP Summary (Influence) - {target_col}")
            shap.summary_plot(shap_values, X_test_current, show=False)
            
        # Cas 2 : Classification (shap_values est une LISTE de tableaux, un par classe)
        else:
            # Pour simplifier la vue, on regarde l'influence sur la classe "Positive" ou la plus sévère
            # Si binaire (0,1) -> On prend l'index 1
            # Si multiclasse (0,1,2,3) -> On prend la dernière classe (souvent la plus grave)
            
            if isinstance(shap_values, list):
                class_index = len(shap_values) - 1 # Prend la dernière classe
                print(f"    [Info] Affichage SHAP pour la classe index {class_index} (Probablement la plus sévère)")
                plt.title(f"SHAP Summary - {target_col} (Classe {class_index})")
                shap.summary_plot(shap_values[class_index], X_test_current, show=False)
            else:
                # Certaines versions récentes de SHAP renvoient un tableau unique pour le binaire
                shap.summary_plot(shap_values, X_test_current, show=False)
                
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"    [Attention] Impossible de générer SHAP : {e}")

    # C. Prédiction et Évaluation (Code original conservé)
    pred_test = model.predict(X_test_current)
    
    if model_type == 'regression':
        pred_test_rounded = np.clip(np.round(pred_test), 0, 3).astype(int)
        acc = accuracy_score(y_test_step, pred_test_rounded)
        r2 = r2_score(y_test_step, pred_test)
        print(f"    -> R2 Score : {r2:.4f}")
        print(f"    -> Accuracy (arrondie) : {acc:.2%}")
        
    else:
        acc = accuracy_score(y_test_step, pred_test)
        f1 = f1_score(y_test_step, pred_test, average='weighted', zero_division=0)
        print(f"    -> Accuracy : {acc:.2%}")
        print(f"    -> F1-Score Weighted : {f1:.4f}")

    # (Code de chaînage)
    # feature_name = f"pred_{target_col}"
    # X_train_current[feature_name] = model.predict(X_train_current)
    # X_test_current[feature_name] = pred_test

print("\n--- Chaîne Terminée ---")