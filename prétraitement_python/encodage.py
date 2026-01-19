# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 16:59:46 2025

@author: Lenovo
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import sys

# --- 1. CONFIGURATION ---
INPUT_CSV = 'runsafe_donnees_finales.csv'  # Le fichier fusionné de l'étape précédente
OUTPUT_CSV = 'runsafe_ml_ready.csv'        # Le fichier final, prêt pour le ML
ENCODER_PATH = 'ml_encoders/'              # Dossier pour sauvegarder les encodeurs

# --- 2. DÉFINITION DES COLONNES ---
# (Ajustez ces listes si nécessaire)

# Variables à prédire (cibles Y)
TARGET_COLUMNS = [
    'InjDefn', 'InjJoint', 
    'InjJoint2'
    # L'utilisateur a mentionné 6, mais il y en a 7 ici. 
    # 'InjDuration' est traité comme une feature numérique.
]

# Variables catégorielles à expliquer (features X)
CATEGORICAL_FEATURES = ['Gender']

# Variables à ignorer pour l'entraînement
IDENTIFIER_COLUMNS = ['sub_id', 'session_file']

print("--- Démarrage du script de préparation ML ---")
print(f"Fichier d'entrée: {INPUT_CSV}")

# --- 3. CHARGEMENT DES DONNÉES ---
try:
    df = pd.read_csv(INPUT_CSV)
    print(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes.")
except FileNotFoundError:
    print(f"ERREUR: Fichier d'entrée '{INPUT_CSV}' introuvable.")
    sys.exit()

df = df.drop(['InjSide', 'InjSide2','SpecInjury', 'SpecInjury2'], axis= 1)

# Créer le dossier pour les encodeurs s'il n'existe pas
import os
os.makedirs(ENCODER_PATH, exist_ok=True)
print(f"Encodeurs et scalers seront sauvegardés dans: {ENCODER_PATH}")

# --- 4. ÉTAPE A: CALCUL DES NOUVELLES VARIABLES (FEATURE ENGINEERING) ---
print("Calcul des variables de force...")
g = 9.81

# S'assurer que les colonnes nécessaires sont numériques
force_cols_num = [
    'Weight', 'L_VERTICAL_OSCILLATION_median', 'L_STANCE_TIME_median',
    'R_VERTICAL_OSCILLATION_median', 'R_STANCE_TIME_median',
    'L_STRIDE_RATE_median', 'R_STRIDE_RATE_median', 'R_VERTICAL_OSCILLATION_std', 'L_VERTICAL_OSCILLATION_std'
]
for col in force_cols_num:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remplacer les NaN par la médiane de la colonne (pour éviter les erreurs de calcul)
for col in force_cols_num:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# 1. Calcul Force d'Impact Médiane (par jambe)
ov_median_L_m = df['L_VERTICAL_OSCILLATION_median'] / 1000.0
v_verticale_L = np.sqrt(2 * g * ov_median_L_m)
df['F_impact_median_L'] = (df['Weight'] * g) + (df['Weight'] * v_verticale_L) / df['L_STANCE_TIME_median']

ov_median_R_m = df['R_VERTICAL_OSCILLATION_median'] / 1000.0
v_verticale_R = np.sqrt(2 * g * ov_median_R_m)
df['F_impact_median_R'] = (df['Weight'] * g) + (df['Weight'] * v_verticale_R) / df['R_STANCE_TIME_median']

# 2. Calcul Taux de Charge Cumulé (Loading Rate)
df['F_impact_median_Avg'] = (df['F_impact_median_L'] + df['F_impact_median_R']) / 2
df['Cadence_strides_min_Avg'] = (df['L_STRIDE_RATE_median'] + df['R_STRIDE_RATE_median']) / 2
df['Cadence_steps_sec_Avg'] = (df['Cadence_strides_min_Avg'] * 2) / 60
df['Loading_Rate_N_per_sec'] = df['F_impact_median_Avg'] * df['Cadence_steps_sec_Avg']
print("Calcul des forces terminé.")

# 1. Définir les 12 noms de variables de base
base_variables = [
    'STRIDE_RATE', 'STRIDE_LENGTH', 'SWING_TIME', 'STANCE_TIME',
    'ANKLE_DF_PEAK_ANGLE', 'ANKLE_EVE_PEAK_ANGLE', 'ANKLE_ROT_PEAK_ANGLE',
    'FOOT_PROG_ANGLE', 'FOOT_ANG_at_HS', 'VERTICAL_OSCILLATION',
    'PRONATION_ONSET', 'PRONATION_OFFSET'
]

# Liste pour garder en mémoire les colonnes originales à supprimer
original_cols_to_drop = []

print("Début de la création des features 'Moyenne' et 'Asymétrie'...")

# 2. Boucler sur chaque variable de base
for base in base_variables:
    
    # --- A. Traitement des paires MEDIAN ---
    
    # Définir les noms des colonnes originales
    col_L_med = f'L_{base}_median'
    col_R_med = f'R_{base}_median'
    
    # Ajouter à la liste de suppression
    original_cols_to_drop.extend([col_L_med, col_R_med])
    
    # Définir les noms des nouvelles colonnes
    new_col_avg = f'{base}_median_Avg'
    new_col_asym = f'{base}_median_Asymmetry'
    
    # Calculer la Moyenne
    df[new_col_avg] = (df[col_L_med] + df[col_R_med]) / 2
    
    # Calculer l'Indice d'Asymétrie (SI) en %
    # Formule: (Droite - Gauche) / (0.5 * (Droite + Gauche)) * 100
    numerator = df[col_R_med] - df[col_L_med]
    denominator = 0.5 * (df[col_R_med] + df[col_L_med])
    
    # Utiliser np.where pour éviter la division par zéro
    # Si le dénominateur est 0, l'asymétrie est 0 (car D et G sont 0)
    df[new_col_asym] = np.where(
        denominator == 0,
        0.0,
        (numerator / denominator) * 100
    )
    
    # Remplacer les NaN potentiels (si D+G != 0 mais D ou G est NaN) par 0
    df[new_col_asym].fillna(0, inplace=True)

    
    # --- B. Traitement des paires STD (Écart-Type) ---
    
    # Définir les noms des colonnes originales
    col_L_std = f'L_{base}_std'
    col_R_std = f'R_{base}_std'
    
    # Ajouter à la liste de suppression
    original_cols_to_drop.extend([col_L_std, col_R_std])
    
    # Définir les noms des nouvelles colonnes
    new_col_avg_std = f'{base}_std_Avg'
    new_col_asym_std = f'{base}_std_Asymmetry' # Asymétrie de la variabilité
    
    # Calculer la Moyenne des écarts-types
    df[new_col_avg_std] = (df[col_L_std] + df[col_R_std]) / 2
    
    # Calculer l'Asymétrie des écarts-types
    numerator_std = df[col_R_std] - df[col_L_std]
    denominator_std = 0.5 * (df[col_R_std] + df[col_L_std])
    
    # Utiliser np.where pour éviter la division par zéro
    df[new_col_asym_std] = np.where(
        denominator_std == 0,
        0.0,
        (numerator_std / denominator_std) * 100
    )
    df[new_col_asym_std].fillna(0, inplace=True)

print(f"Création de {len(base_variables) * 4} nouvelles features terminée.")

# 3. Supprimer les 48 colonnes L/R originales
# S'assurer que les colonnes existent avant de les supprimer (évite les erreurs)
existing_cols_to_drop = [col for col in original_cols_to_drop if col in df.columns]
df = df.drop(columns=existing_cols_to_drop)

print(f"Suppression de {len(existing_cols_to_drop)} colonnes L/R originales.")
print("Transformation terminée.")

# Afficher les nouvelles colonnes pour vérification
print("\nNouvelles colonnes dans le DataFrame :")
print(df.columns.tolist())

# --- 3b. NETTOYAGE SPÉCIFIQUE : Unknown -> No Injury ---
print("\n--- Correction des valeurs 'Unknown' ---")


# --- 5. ÉTAPE B: ENCODAGE ET NORMALISATION ---

# Séparer les identifiants, les cibles (Y) et les features (X)
df_identifiers = df[IDENTIFIER_COLUMNS]
df_targets = df[TARGET_COLUMNS]
df_features = df.drop(columns=TARGET_COLUMNS + IDENTIFIER_COLUMNS)
    
# Colonne source pour la nouvelle cible
DURATION_COLUMN = 'InjDuration'
NEW_TARGET = 'Injured_Chronic'

print(f"Traitement de la variable cible '{DURATION_COLUMN}'...")

# 1. Remplacer les NaN de InjDuration par 0
# S'assurer qu'elle est numérique (au cas où) et remplacer NaN
df[DURATION_COLUMN] = pd.to_numeric(df[DURATION_COLUMN], errors='coerce')
df[DURATION_COLUMN].fillna(0, inplace=True)
print(f"'{DURATION_COLUMN}': NaN remplacés par 0.")

# 2. Créer une nouvelle colonne cible binaire 'Injured_Chronic'
# (1 si InjDuration > 0, sinon 0)
df[NEW_TARGET] = (df[DURATION_COLUMN] > 0).astype(int)
print(f"Nouvelle colonne cible binaire '{NEW_TARGET}' créée.")

# Afficher la distribution (très important)
target_distribution = df[NEW_TARGET].value_counts(normalize=True) * 100
print(f"Distribution de '{NEW_TARGET}':\n{target_distribution}")

if target_distribution.min() < 10:
    print("\nATTENTION: La nouvelle variable cible est très déséquilibrée.")
    print("Pensez à utiliser 'class_weight=\"balanced\"' lors de l'entraînement.\n")

# 3. Mettre à jour la liste des cibles à encoder
# On ajoute la nouvelle cible binaire
TARGET_COLUMNS.append(NEW_TARGET)

# 4. Séparer les dataframes
df_identifiers = df[IDENTIFIER_COLUMNS]
df_targets = df[TARGET_COLUMNS]

# 5. 'InjDuration' (l'original) ne doit plus être une feature
# On la supprime de la liste des features (X)
df_features = df.drop(columns=TARGET_COLUMNS + [DURATION_COLUMN] + IDENTIFIER_COLUMNS)
print(f"'{DURATION_COLUMN}' (originale) supprimée des features pour éviter la fuite de données.")
    
# --- 5a. Encodage des Cibles (Y) ---
print(f"Encodage des {len(TARGET_COLUMNS)} variables cibles (Y)...")
df_targets_encoded = pd.DataFrame()

for col in TARGET_COLUMNS:
    
    # Cas 1 : La nouvelle cible binaire (déjà 0/1)
    if col == NEW_TARGET:
        df_targets_encoded[col + '_encoded'] = df_targets[col]
        print(f"Cible '{col}' déjà numérique (0/1), conservée telle quelle.")
        
    # Cas 2 : InjDefn (Encodage Ordinal FORCÉ via LabelEncoder)
    elif col == 'InjDefn':
        print(f"Encodage manuel ordinal pour '{col}'...")
        
        # 1. Définir l'ordre strict des classes (l'index deviendra la valeur encodée)
        # 0: No Injury, 1: Continuing..., 2: training..., 3: 2 workouts...
        ordered_classes = [
            'No injury', 
            'Continuing to train in pain', 
            'Training volume/intensity affected', 
            '2 workouts missed in a row'
        ]
        
        # 2. Créer le LabelEncoder et forcer ses classes
        le = LabelEncoder()
        le.fit(ordered_classes) # Initialisation basique
        # L'étape CRUCIALE : On écrase l'ordre alphabétique par notre ordre logique
        le.classes_ = np.array(ordered_classes) 
        
        # 3. Transformer les données
        # Note: Cela plantera s'il reste des valeurs non listées (ex: "Unknown"), 
        # mais votre nettoyage précédent (étape 3b) a déjà réglé ça.
        df_targets_encoded[col + '_encoded'] = le.transform(df_targets[col].astype(str))
        
        # 4. Sauvegarder en .joblib comme les autres
        joblib.dump(le, os.path.join(ENCODER_PATH, f'label_encoder_{col}.joblib'))
        print(f"Encodeur ordinal pour '{col}' sauvegardé.")

    # Cas 3 : Les autres variables catégorielles (LabelEncoder standard alphabétique)
    else:
        le = LabelEncoder()
        encoded_col = le.fit_transform(df_targets[col].astype(str))
        df_targets_encoded[col + '_encoded'] = encoded_col
        # Sauvegarder l'encodeur
        joblib.dump(le, os.path.join(ENCODER_PATH, f'label_encoder_{col}.joblib'))

print("Toutes les variables cibles sont encodées et sauvegardées.")


# --- 5b. Encodage des Features (X) ---
print("Encodage des variables explicatives (X)...")

# One-Hot Encoding pour les variables catégorielles
# 'drop_first=True' évite la multicolinéarité (ex: 'Gender_Male' suffit)
df_features_processed = pd.get_dummies(df_features, columns=CATEGORICAL_FEATURES, drop_first=True)
print(f"Variables catégorielles ({CATEGORICAL_FEATURES}) encodées (One-Hot).")

# --- 5c. Normalisation des Features Numériques (X) ---
# Identifier TOUTES les colonnes numériques (y compris les nouvelles)
numerical_features = df_features_processed.select_dtypes(include=np.number).columns.tolist()

print(f"Normalisation (StandardScaler) de {len(numerical_features)} variables numériques...")
scaler = StandardScaler()

# Appliquer le scaler
df_features_processed[numerical_features] = scaler.fit_transform(df_features_processed[numerical_features])

# Sauvegarder le scaler
joblib.dump(scaler, os.path.join(ENCODER_PATH, 'standard_scaler_X.joblib'))
print("Variables numériques normalisées et scaler sauvegardé.")

# --- 6. COMBINER ET SAUVEGARDER LE FICHIER FINAL ---
# Recombiner les identifiants, les X traités, et les Y encodés
df_final_ml = pd.concat([
    df_identifiers,
    df_features_processed,
    df_targets_encoded
], axis=1)

df_final_ml = df_final_ml.drop(['Gender_Unknown','F_impact_median_L', 'F_impact_median_R','F_impact_median_Avg','Cadence_strides_min_Avg', 'Cadence_steps_sec_Avg','STRIDE_LENGTH_std_Asymmetry','STRIDE_LENGTH_median_Asymmetry', 'STANCE_TIME_median_Avg','STANCE_TIME_median_Asymmetry' ], axis = 1)

# Sauvegarder le fichier CSV final
df_final_ml.to_csv(OUTPUT_CSV, index=False)

nb = (df_final_ml['InjDefn_encoded'] == -1).sum()
print(nb)

print("\n--- TERMINÉ ---")
print(f"Fichier de données prêt pour le ML sauvegardé sous: {OUTPUT_CSV}")
print("Ce fichier contient les données entièrement encodées et normalisées.")
print(f"Les encodeurs (LabelEncoder) et le normalisateur (StandardScaler) sont sauvegardés dans le dossier: {ENCODER_PATH}")