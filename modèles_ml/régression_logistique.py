# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 11:10:20 2025

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score, confusion_matrix
import sys
import os

# --- 1. CONFIGURATION ---
INPUT_CSV = 'runsafe_ml_ready.csv'
ENCODER_PATH = 'ml_encoders/'

MODEL_OUTPUT_FILE = 'logreg1_model.joblib'
METRICS_OUTPUT_FILE = 'logreg1_metrics.csv' 

print(f"--- 1. Chargement des données prêtes pour le ML ---")
try:
    df_ml = pd.read_csv(INPUT_CSV)
    print(f"Données chargées: {df_ml.shape[0]} lignes, {df_ml.shape[1]} colonnes.")
except FileNotFoundError:
    print(f"ERREUR: Fichier '{INPUT_CSV}' introuvable. Avez-vous exécuté le script de préparation ?")
    sys.exit()

# --- 2. SÉPARATION DES DONNÉES (X et Y) ---
print("--- 2. Séparation des variables (X et Y) ---")

# Définir les colonnes cibles (Y) - celles qui finissent par '_encoded'
TARGET_COLS = [col for col in df_ml.columns if col.endswith('_encoded')]

# Définir les colonnes d'identification
ID_COLS = ['sub_id', 'session_file']
print('salut')

# Les features (X) sont tout le reste
FEATURE_COLS = [col for col in df_ml.columns if col not in TARGET_COLS and col not in ID_COLS]

X = df_ml[FEATURE_COLS]
y = df_ml[TARGET_COLS]

print(f"{len(FEATURE_COLS)} variables explicatives (X) identifiées.")
print(f"{len(TARGET_COLS)} variables cibles (Y) identifiées: {TARGET_COLS}")

# --- 3. CRÉATION DES ENSEMBLES D'ENTRAÎNEMENT ET DE TEST ---
print("\n--- 3. Création des ensembles d'entraînement et de test ---")
# 80% pour l'entraînement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Taille de l'ensemble d'entraînement: {X_train.shape[0]} échantillons")
print(f"Taille de l'ensemble de test: {X_test.shape[0]} échantillons")

# --- 4. CONFIGURATION DU MODÈLE ---
print("\n--- 4. Configuration de la Régression Logistique Multi-Sorties ---")

# 4a. Définir le modèle de base (un seul LogisticRegression)
# Nous configurons ce modèle pour qu'il soit performant :
base_model = SVC(
    kernel='rbf',
    class_weight='balanced',  
    C=1.0,                    
    gamma='scale',            
    probability=True,         
    random_state=42,
    cache_size=1000
)

# 4b. Créer l'enveloppe Multi-Sorties
# n_jobs=-1 utilise tous les cœurs du CPU pour entraîner les 6 modèles en parallèle
multi_output_model = MultiOutputClassifier(base_model, n_jobs=-1)

print("Modèle de base (LogisticRegression) configuré avec 'class_weight=balanced'.")
print("Modèle 'MultiOutputClassifier' prêt.")

# --- 5. ENTRAÎNEMENT DU MODÈLE ---
print("\n--- 5. Entraînement en cours... (cela peut prendre un moment) ---")
multi_output_model.fit(X_train, y_train)
print("Entraînement terminé.")
joblib.dump(multi_output_model, MODEL_OUTPUT_FILE)
print(f"Modèle sauvegardé sous: {MODEL_OUTPUT_FILE}")

# --- 6. ÉVALUATION DU MODÈLE ---
print("\n--- 6. Évaluation sur l'ensemble de test ---")
y_pred = multi_output_model.predict(X_test)
y_probas_list = multi_output_model.predict_proba(X_test)

# Convertir y_test (DataFrame) et y_pred (ndarray) en entiers pour la comparaison
y_test_values = y_test.values

model_metrics = []

# Évaluation cible par cible
for i, col_name in enumerate(TARGET_COLS):
    print(f"\n--- Résultats pour la cible : {col_name} ---")
    
    # Extraire les encodeurs de labels pour afficher les noms des classes
    # Récupérer les données pour CETTE cible spécifique
    y_true_col = y_test_values[:, i]
    y_prob_col = y_probas_list[i] # Les probas pour cette cible
    
    # --- CALCUL DE L'AUC-ROC SPÉCIFIQUE ---
    try:
        # Cas 1 : Binaire (ex: Injured_Chronic)
        if y_prob_col.shape[1] == 2:
            # On prend la proba de la classe positive (colonne 1)
            auc = roc_auc_score(y_true_col, y_prob_col[:, 1])
            
        # Cas 2 : Multiclasse (ex: InjJoint)
        else:
            # On passe toutes les probas et on précise 'ovr' (One-vs-Rest)
            auc = roc_auc_score(y_true_col, y_prob_col, multi_class='ovr', average='macro')
            
        print(f"AUC-ROC : {auc:.4f}")
        
    except ValueError as e:
        print(f"Impossible de calculer l'AUC (peut-être une seule classe présente ?) : {e}")
        auc = None
    try:
        if col_name != 'Injured_Chronic_encoded': # Cible binaire n'a pas d'encodeur
            encoder_file = os.path.join(ENCODER_PATH, f'label_encoder_{col_name.replace("_encoded", "")}.joblib')
            le = joblib.load(encoder_file)
            target_names = le.classes_
        else:
            target_names = ['Non_Chronic', 'Chronic'] # Noms pour la cible binaire
            
        print(classification_report(y_test_values[:, i], y_pred[:, i], target_names=target_names, zero_division=0))
        
        report_dict = classification_report(y_test_values[:, i], y_pred[:, i], output_dict=True, zero_division=0)
        
        metrics_dict = {
            'target': col_name.replace("_encoded", ""),
            'accuracy': report_dict['accuracy'],
            'f1_weighted': report_dict['weighted avg']['f1-score'],
            'precision_weighted': report_dict['weighted avg']['precision'],
            'recall_weighted': report_dict['weighted avg']['recall']
        }
        model_metrics.append(metrics_dict)
    except FileNotFoundError:
        print(f"Rapport pour {col_name}:")
        print(classification_report(y_test_values[:, i], y_pred[:, i], zero_division=0))
    except Exception as e:
        print(f"Erreur lors de la génération du rapport pour {col_name}: {e}")


if model_metrics:
    df_metrics = pd.DataFrame(model_metrics)
    df_metrics.to_csv(METRICS_OUTPUT_FILE, index=False)
    print("\n--- Métriques de performance exportées ---")
    print(f"Tableau récapitulatif sauvegardé sous: {METRICS_OUTPUT_FILE}")
    print(df_metrics)
    
for TARGET_NAME in TARGET_COLS:
    # 1. Trouver l'index de la colonne InjDefn dans les prédictions
    idx = TARGET_COLS.index(TARGET_NAME)
    
    # 2. Extraire les vecteurs correspondants (Réel et Prédit)
    # y_test_values et y_pred sont déjà définis plus haut dans le script (Section 6)
    y_true_specific = y_test_values[:, idx]
    y_pred_specific = y_pred[:, idx]

    # --- A. Matrice de Confusion ---
    cm = confusion_matrix(y_true_specific, y_pred_specific)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de Confusion - {TARGET_NAME}')
    plt.xlabel('Prédiction')
    plt.ylabel('Valeur Réelle')
    plt.show()

    # --- B. Histogramme des erreurs ---
    # Comme InjDefn est ordinal (0, 1, 2, 3), la différence mathématique a du sens
    # Erreur = 0 (Parfait), -1 (Sous-estimé de 1 niveau), +2 (Surestimé de 2 niveaux)...
    errors = y_pred_specific - y_true_specific
    
    plt.figure(figsize=(8, 4))
    # 'discrete=True' rend l'histogramme plus propre pour des entiers
    sns.histplot(errors, kde=True, discrete=True) 
    plt.title(f"Distribution des Erreurs pour {TARGET_NAME}\n(Prédit - Réel)")
    plt.xlabel("Erreur (0 = Prédiction parfaite)")
    plt.ylabel("Nombre d'échantillons")
    plt.axvline(0, color='red', linestyle='--', linewidth=1) # Ligne repère à 0
    plt.show()

# --- 7. EXEMPLE DE PRÉDICTION INDIVIDUELLE ---
print("\n--- 7. Exemple de prédiction sur un seul échantillon ---")
# Prendre le premier échantillon de l'ensemble de test
sample = X_test.iloc[0:1] 
prediction_encoded = multi_output_model.predict(sample)

print(f"Prédiction (encodée): {prediction_encoded}")

# Re-traduire la prédiction en texte
prediction_readable = {}
for i, col_name in enumerate(TARGET_COLS):
    col_root = col_name.replace("_encoded", "")
    
    # Gérer la cible binaire manuellement
    if col_root == 'Injured_Chronic':
        prediction_readable[col_root] = 'Oui' if prediction_encoded[0, i] == 1 else 'Non'
    else:
        # Charger l'encodeur
        try:
            encoder_file = os.path.join(ENCODER_PATH, f'label_encoder_{col_root}.joblib')
            le = joblib.load(encoder_file)
            # Utiliser inverse_transform pour retrouver le texte
            readable_label = le.inverse_transform([prediction_encoded[0, i]])
            prediction_readable[col_root] = readable_label[0]
        except:
            prediction_readable[col_root] = f"Erreur_Encodage (val={prediction_encoded[0, i]})"

print("\nPrédiction (lisible):")
print(prediction_readable)

