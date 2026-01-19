# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 18:19:29 2025

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

# --- 1. CONFIGURATION ---
INPUT_CSV = 'runsafe_ml_ready.csv'

print("--- Chaîne Hybride Personnalisée (4 Variables) ---")
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
# On exclut TOUTES les cibles potentielles du fichier (même celles qu'on ne prédit pas ici)
# pour éviter la fuite de données (data leakage).
all_encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
identifiers = ['sub_id', 'session_file']
cols_to_drop = all_encoded_cols + identifiers

# Définition de X (Features) et y (Targets)
X = df.drop(columns=cols_to_drop)
y = df[ordered_targets]

print(f"Features (X) : {X.shape[1]} variables")
print(f"Cibles (y) : {ordered_targets}")

# Split (Même random_state pour comparer)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. DÉFINITION DE LA CHAÎNE (ORDRE ET MODÈLES) ---

chain_steps = [
    {
        'target': 'Injured_Chronic_encoded', 
        'type': 'classification',
        'model': SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    },
    {
        'target': 'InjDefn_encoded', 
        'type': 'classification',
        'model': SVC(
            kernel='rbf',    
            class_weight='balanced', 
            probability=True,        
            random_state=42     
        )
    },
    {
        'target': 'InjJoint_encoded', 
        'type': 'classification',
        'model': SVC(
            kernel='rbf',
            class_weight='balanced', 
            probability=True,        
            random_state=42
        )
    },
    {
        'target': 'InjJoint2_encoded', 
        'type': 'classification',
        'model': SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    }
]

# --- 4. BOUCLE D'ENTRAÎNEMENT ET DE PRÉDICTION ---

# Copies de travail pour ajouter les prédictions au fur et à mesure
X_train_current = X_train.copy()
X_test_current = X_test.copy()

print("\n--- Début de la Chaîne ---")

for i, step in enumerate(chain_steps):
    target_col = step['target']
    model_type = step['type']
    model = step['model']
    
    print(f"\nÉtape {i+1}: Traitement de {target_col} ({model_type})")
    
    # A. Sélectionner la cible unique pour cette étape
    y_train_step = y_train[target_col]
    y_test_step = y_test[target_col]
    
    # B. Entraîner le modèle sur les données courantes (X + prédictions précédentes)
    model.fit(X_train_current, y_train_step)
    
    # C. Prédire
    pred_train = model.predict(X_train_current)
    pred_test = model.predict(X_test_current)
    
    # D. Évaluation
    if model_type == 'regression':
        # Pour la régression (InjDefn), on arrondit pour calculer une "précision" indicative
        # tout en gardant le score R2 pour la qualité réelle
        pred_test_rounded = np.clip(np.round(pred_test), 0, 3).astype(int)
        acc = accuracy_score(y_test_step, pred_test_rounded)
        r2 = r2_score(y_test_step, pred_test)
        print(f"    -> R2 Score : {r2:.4f}")
        print(f"    -> Accuracy (arrondie) : {acc:.2%}")
        print("\n    -> Rapport détaillé (sur valeurs arrondies) :")
        print(classification_report(y_test_step, pred_test_rounded, zero_division=0))
        
    else:
        # Pour la classification
        acc = accuracy_score(y_test_step, pred_test)
        f1 = f1_score(y_test_step, pred_test, average='weighted', zero_division=0)
        try:
            # On doit calculer les PROBABILITÉS pour l'AUC
            y_prob = model.predict_proba(X_test_current)
            
            # Gestion automatique Binaire vs Multiclasse
            if y_prob.shape[1] == 2:
                # Cas binaire : on donne juste la proba de la classe 1
                auc_score = roc_auc_score(y_test_step, y_prob[:, 1])
            else:
                # Cas multiclasse : on donne toute la matrice
                auc_score = roc_auc_score(y_test_step, y_prob, multi_class='ovr', average='macro')
                
            print(f"    -> Accuracy          : {acc:.2%}")
            print(f"    -> F1-Score Weighted : {f1:.4f}")
            print(f"    -> AUC-ROC           : {auc_score:.4f}")
            
        except Exception as e:
            print(f"    -> AUC-ROC           : Erreur calcul ({e})")

        # print("\n    -> Rapport détaillé :")
        # print(classification_report(y_test_step, pred_test, zero_division=0)
        # auc_score = roc_auc_score(y_test_step, pred_test, multi_class='ovr', average='macro')
        
        # print(f"    -> Accuracy            : {acc:.2%}")
        # print(f"    -> F1-Score Weighted   : {f1:.4f}")
        # print(f"    -> AUC-score            : {auc_score:.4f}")
        
        # print("\n    -> Rapport détaillé :")
        # # Le rapport affiche Précision, Rappel et F1 pour CHAQUE classe
        # print(classification_report(y_test_step, pred_test, zero_division=0))
        
    # E. AJOUT DE LA PRÉDICTION POUR L'ÉTAPE SUIVANTE
    # C'est la clé de la chaîne : la prédiction devient une feature
    feature_name = f"pred_{target_col}"
    
    # Note : Si c'est une régression, on injecte la valeur continue (plus riche).
    # Si c'est une classification, on injecte la classe prédite.
    X_train_current[feature_name] = pred_train
    X_test_current[feature_name] = pred_test
    
    print(f"    [Info] Prédiction '{feature_name}' ajoutée aux features pour la suite.")

print("\n--- Chaîne Terminée ---")