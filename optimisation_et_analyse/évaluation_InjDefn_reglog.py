# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 18:50:42 2025

@author: Lenovo
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_auc_score

# --- 1. CONFIGURATION ---
INPUT_CSV = 'runsafe_ml_ready.csv'
ENCODER_PATH = 'ml_encoders/' 
MODEL_FILE = 'logreg1_model.joblib' # Assurez-vous que le modèle est sauvegardé

print("--- 1. Chargement des données et du modèle ---")
try:
    df_ml = pd.read_csv(INPUT_CSV)
    multi_output_model = joblib.load(MODEL_FILE)
    print(f"Données et modèle '{MODEL_FILE}' chargés.")
except FileNotFoundError as e:
    print(f"ERREUR: Fichier introuvable. {e}")
    sys.exit()

# --- 2. SÉPARATION DES DONNÉES (X et Y) ---
TARGET_COLS = [col for col in df_ml.columns if col.endswith('_encoded')]
ID_COLS = ['sub_id', 'session_file']
FEATURE_COLS = [col for col in df_ml.columns if col not in TARGET_COLS and col not in ID_COLS]
X = df_ml[FEATURE_COLS]
y = df_ml[TARGET_COLS]

# --- 3. CRÉATION DES ENSEMBLES D'ENTRAÎNEMENT ET DE TEST (identique à l'entraînement) ---
# Il est crucial d'utiliser le même 'random_state' pour avoir les mêmes données de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Ensemble de test recréé ({len(X_test)} échantillons).")

# --- 4. ÉVALUATION AVANCÉE POUR 'InjDefn' ---
print("\n--- 4. Évaluation avancée pour 'InjDefn' ---")

# 4a. Isoler la cible 'InjDefn'
TARGET_NAME = 'InjDefn'
TARGET_COL_ENCODED = TARGET_NAME + '_encoded'

try:
    target_index = TARGET_COLS.index(TARGET_COL_ENCODED)
    # Récupérer le modèle spécifique pour 'InjDefn'
    model_injdefn = multi_output_model.estimators_[target_index]
    
    # Charger l'encodeur pour avoir les noms des classes
    le = joblib.load(os.path.join(ENCODER_PATH, f'label_encoder_{TARGET_NAME}.joblib'))
    class_names = le.classes_
    
    # Obtenir les valeurs réelles et les prédictions
    y_true = y_test.values[:, target_index]
    y_pred = model_injdefn.predict(X_test)
    y_prob = model_injdefn.predict_proba(X_test) # Probabilités pour l'AUC

except (ValueError, FileNotFoundError):
    print(f"ERREUR: Impossible de trouver la cible '{TARGET_COL_ENCODED}' ou son encodeur.")
    sys.exit()

# 4b. Afficher le Rapport de Classification (Précision, Rappel, F1)
print("\n--- Rapport de Classification (F1, Rappel, Précision) ---")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# 4c. Calculer l'AUC-ROC (Multiclasse, One-vs-Rest)
# 'average="macro"' : Moyenne simple des scores OvR. Bon pour le déséquilibre.
auc_score = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
print("\n--- Score AUC-ROC (Multiclasse, OvR, Moyenne Macro) ---")
print(f"AUC = {auc_score:.4f}")
print("(Plus c'est proche de 1.0, mieux le modèle discrimine les sévérités)")

# 4d. Calculer et afficher la Matrice de Confusion
print("\n--- Matrice de Confusion ---")
cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

# Créer la heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matrice de Confusion pour {TARGET_NAME}')
plt.xlabel('Prédiction')
plt.ylabel('Valeur Réelle')

# Sauvegarder l'image
output_image_path = 'matrice_confusion_injdefn.png'
plt.savefig(output_image_path)
print(f"Matrice de confusion sauvegardée sous: {output_image_path}")
plt.show()