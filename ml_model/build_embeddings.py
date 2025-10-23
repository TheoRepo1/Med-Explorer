import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

CLEANED_DATA_PATH = os.path.join('data', 'cleaned_sempex.csv')
EMBEDDINGS_PATH = os.path.join('data', 'embeddings.npy')

def build_and_save_embeddings():
    """
    Charge les données nettoyées, crée des phrases descriptives,
    les encode en vecteurs (embeddings) et les sauvegarde.
    """
    print("Chargement des données nettoyées pour la création des embeddings...")
    if not os.path.exists(CLEANED_DATA_PATH):
        print(f"ERREUR: Fichier {CLEANED_DATA_PATH} non trouvé. Lancez d'abord src/cleaning.py")
        return

    df = pd.read_csv(CLEANED_DATA_PATH)
    
    # AMÉLIORATION : On crée la description SANS la marque pour recentrer l'IA
    df['description'] = df.apply(
        lambda row: f"{row['DCI']} {row['dosage']} {row['forme']}".replace('nan', ''),
        axis=1
    )
    
    print("Chargement du modèle d'embedding léger et rapide...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print(f"Création des embeddings pour {len(df)} médicaments...")
    embeddings = model.encode(df['description'].tolist(), show_progress_bar=True)
    
    print(f"Sauvegarde des embeddings dans {EMBEDDINGS_PATH}...")
    np.save(EMBEDDINGS_PATH, embeddings)
    
    print("✅ Embeddings créés et sauvegardés avec succès.")

if __name__ == "__main__":
    build_and_save_embeddings()