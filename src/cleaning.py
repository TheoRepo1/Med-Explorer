import pandas as pd
import os
from ml_model.feature_extractor import feature_extractor_instance

RAW_DATA_PATH = os.path.join('data', 'sempex.csv')
CLEANED_DATA_PATH = os.path.join('data', 'cleaned_sempex.csv')

def clean_data():
    """
    Lit les données brutes, les nettoie, extrait les caractéristiques
    et les sauvegarde dans un nouveau fichier CSV.
    """
    print("Début du nettoyage et de l'enrichissement des données...")

    try:
        df = pd.read_csv(RAW_DATA_PATH, dtype={'cip7': str})
    except FileNotFoundError:
        print(f"ERREUR: Le fichier {RAW_DATA_PATH} est introuvable.")
        return

    # 1. Sélection et renommage des colonnes
    columns_to_keep = [
        'libelle_long', 'libelle_court', 'dci', 'prix_brousse', 
        'prix_iles', 'date_application', 'code_remboursement'
    ]
    df = df.reindex(columns=columns_to_keep)
    df = df.rename(columns={
        'libelle_long': 'Nom Complet', 'libelle_court': 'Nom Court', 'dci': 'DCI',
        'prix_brousse': 'Prix Brousse', 'prix_iles': 'Prix Nouméa',
        'date_application': 'Date Application', 'code_remboursement': 'Code Remboursement'
    })

    # 2. Conversion des types et gestion des valeurs manquantes
    price_cols = ['Prix Brousse', 'Prix Nouméa']
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    remboursement_map = {3: '65%', 0: '0%', 5: '100%'}
    df['Taux Remboursement'] = df['Code Remboursement'].map(remboursement_map).fillna('Non défini')
    df['DCI'] = df['DCI'].fillna('Non spécifiée')

    # 3. Extraction par lots des caractéristiques
    print("Extraction des caractéristiques avec spaCy (méthode par lots)...")
    
    extracted_data = feature_extractor_instance.extract_batch(df['Nom Complet'])
    features_df = pd.DataFrame(extracted_data)
    
    df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

    # 4. Sauvegarder le fichier nettoyé et enrichi
    os.makedirs('data', exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    
    print(f"Nettoyage et enrichissement terminés. Fichier sauvegardé dans {CLEANED_DATA_PATH}")

if __name__ == "__main__":
    clean_data()