# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 13:48:03 2025

@author: Lenovo
"""

import pandas as pd
import numpy as np

# --- 1. CONFIGURATION ---
# Chemin vers votre fichier CSV principal
csv_path = 'C:/Users\Lenovo/OneDrive - Early Makers Group/Documents/Etudes/EMSE/RunSafe/Indus algo/run_data_meta.csv' 
# Nom du fichier CSV de sortie final
output_csv_path = 'C:/Users\Lenovo/OneDrive - Early Makers Group/Documents/Etudes/EMSE/RunSafe/Indus algo/runsafe_donnees_pretraitees_final.csv'


# --- 2. FILTRAGE INITIAL DES DONNÉES SUJET ---
columns_to_keep = [
    'sub_id', 'filename', 'speed_r', 'age', 'Height', 'Weight', 'Gender', 'YrsRunning',
    'InjDefn', 'InjJoint', 'InjSide', 'SpecInjury', 'InjDuration', 
    'InjJoint2', 'InjSide2', 'SpecInjury2'
]

try:
    df_meta = pd.read_csv(csv_path)
    df_meta = df_meta[columns_to_keep].copy()
except FileNotFoundError:
    print(f"Erreur : Le fichier {csv_path} est introuvable.")
    exit()
except KeyError as e:
    print(f"Erreur : Colonne manquante {e} dans le fichier CSV. Vérifiez les en-têtes.")
    exit()

# Afficher les premières lignes et les types de données pour vérification
print("\nPremières lignes des métadonnées sélectionnées:")
print(df_meta.head())
print("\nTypes de données:")
print(df_meta.info())

numerical_cols = ['speed_r', 'age', 'Height', 'Weight', 'YrsRunning']

print("\n--- Traitement des outliers numériques ---")

for col in numerical_cols:
    if col in df_meta.columns:
        # Assurer que la colonne est numérique, convertir si possible
        df_meta[col] = pd.to_numeric(df_meta[col], errors='coerce') 
        
        # 1. Vérifications basiques/logiques
        original_nan_count = df_meta[col].isnull().sum()
        if col == 'age':
            df_meta.loc[df_meta[col] < 12, col] = np.nan # Âge minimum
            df_meta.loc[df_meta[col] > 90, col] = np.nan # Âge maximum
        elif col == 'Height':
            df_meta.loc[df_meta[col] < 100, col] = np.nan # Taille min (cm)
            df_meta.loc[df_meta[col] > 250, col] = np.nan # Taille max (cm)
        elif col == 'Weight':
            df_meta.loc[df_meta[col] < 30, col] = np.nan  # Poids min (kg)
            df_meta.loc[df_meta[col] > 200, col] = np.nan # Poids max (kg)
        elif col in ['YrsRunning', 'InjDuration']:
             df_meta.loc[df_meta[col] < 0, col] = np.nan # Ne peut pas être négatif
        # Ajoutez d'autres règles si nécessaire pour speed_r

        logical_nan_count = df_meta[col].isnull().sum() - original_nan_count
        if logical_nan_count > 0:
             print(f"'{col}': {logical_nan_count} valeurs remplacées par NaN (vérification logique).")

        # 2. Détection statistique (IQR)
        Q1 = df_meta[col].quantile(0.25)
        Q3 = df_meta[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identifier les outliers
        outliers_mask = (df_meta[col] < lower_bound) | (df_meta[col] > upper_bound)
        num_outliers = outliers_mask.sum()
        
        if num_outliers > 0:
            # Remplacer les outliers par NaN pour l'instant (ou par la médiane/moyenne si préféré)
            median_val = df_meta[col].median() # Calculer la médiane AVANT de remplacer
            df_meta.loc[outliers_mask, col] = median_val # Remplacement par la médiane
            print(f"'{col}': {num_outliers} outliers statistiques détectés (IQR) et remplacés par la médiane ({median_val:.2f}).")
            
        # Optionnel: Remplacer les NaN restants (après coercition, logique, IQR) par la médiane
        final_nan_count = df_meta[col].isnull().sum()
        if final_nan_count > 0:
             median_final = df_meta[col].median() # Recalculer la médiane sans les outliers
             df_meta[col].fillna(median_final, inplace=True)
             print(f"'{col}': {final_nan_count} NaN restants remplacés par la médiane ({median_final:.2f}).")

print("\nDescription après nettoyage numérique:")
print(df_meta[numerical_cols].describe())

categorical_cols = ['Gender', 'InjDefn', 'InjJoint', 'InjSide', 'SpecInjury', 
                    'InjJoint2', 'InjSide2', 'SpecInjury2']

print("\n--- Traitement des variables catégorielles ---")

for col in categorical_cols:
    if col in df_meta.columns:
        # Remplacer les valeurs manquantes (souvent vides ou NaN) par 'Unknown' ou 'No Injury'
        # Convertir en string pour faciliter le traitement
        df_meta[col] = df_meta[col].astype(str).fillna('No injury') 
        df_meta[col].replace(['nan', '', ' '], 'No injury', inplace=True) # Remplacer les NaN textuels ou vides

df_meta_cleaned = df_meta # Renommer pour la clarté


# Chemin vers le fichier CSV généré par MATLAB
biomechanics_csv_path = 'runsafe_processed_data.csv' 
# Chemin pour le fichier CSV final fusionné
output_csv_path = 'runsafe_donnees_finales.csv'

try:
    df_biomechanics = pd.read_csv(biomechanics_csv_path)
    print(f"\nChargement des données biomécaniques réussi ({len(df_biomechanics)} sessions).")
except FileNotFoundError:
    print(f"Erreur: Le fichier {biomechanics_csv_path} est introuvable.")
    exit()


    

# Fusionner les deux DataFrames
# On joint les métadonnées nettoyées (df_meta_cleaned) sur la table des sessions (df_biomechanics)
df_final = pd.merge(
    df_biomechanics,    # DataFrame de gauche (données par session)
    df_meta_cleaned,    # DataFrame de droite (métadonnées nettoyées par sujet)
    left_on=['session_file','sub_id'],        # Clé de jointure
    right_on=['filename','sub_id'],
    how='left'          # Garder toutes les sessions, ajouter les métadonnées correspondantes
)

df_final = df_final.drop(['filename'], axis=1)

# Vérifier la fusion
print(f"\nFusion terminée. Le DataFrame final contient {len(df_final)} lignes et {len(df_final.columns)} colonnes.")
print("Colonnes après fusion:", df_final.columns.tolist())
print(df_final.head())


# --- 4. Sauvegarde du Fichier Final ---
df_final.to_csv(output_csv_path, index=False, float_format='%.6f')
print(f"\nDataFrame final sauvegardé sous: {output_csv_path}")