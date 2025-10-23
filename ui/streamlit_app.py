import pandas as pd
import streamlit as st
import os
import numpy as np
from sentence_transformers import util
import altair as alt

st.set_page_config(layout="wide", page_title="Med-Explorer", page_icon="💊")

@st.cache_data
def load_all_data():
    """
    Charge le dataframe des médicaments et la matrice des embeddings depuis les fichiers locaux.
    """
    data_path = os.path.join('data', 'cleaned_sempex.csv')
    embeddings_path = os.path.join('data', 'embeddings.npy')
    
    if not os.path.exists(data_path) or not os.path.exists(embeddings_path):
        st.error("Fichiers de données (cleaned_sempex.csv ou embeddings.npy) non trouvés. Assurez-vous qu'ils sont bien dans le dépôt.")
        return None, None
        
    df = pd.read_csv(data_path)
    df['Taux Remboursement'] = df['Taux Remboursement'].astype('category')
    embeddings = np.load(embeddings_path)
    
    return df, embeddings

data, embeddings = load_all_data()

# --- Panneau Latéral (Sidebar) ---
st.sidebar.header("🔍 Filtres")

if data is None or embeddings is None:
    # Le message d'erreur est déjà affiché dans la fonction de chargement
    st.stop()
else:
    # Filtre sur le taux de remboursement
    taux_disponibles = data['Taux Remboursement'].unique().tolist()
    taux_selectionnes = st.sidebar.multiselect(
        "Taux de Remboursement", 
        options=taux_disponibles, 
        default=taux_disponibles
    )
    
    # Filtre sur la fourchette de prix
    valid_prices = data['Prix Nouméa'].dropna()
    if not valid_prices.empty:
        min_price, max_price = int(valid_prices.min()), int(valid_prices.max())
        prix_selectionnes = st.sidebar.slider(
            "Fourchette de Prix (Nouméa)", 
            min_price, 
            max_price, 
            (min_price, max_price)
        )
    else:
        prix_selectionnes = (0, 0)
        
    if st.sidebar.button("Réinitialiser les filtres"): 
        st.rerun()

    # Application des filtres de la sidebar
    price_condition = (data['Prix Nouméa'].between(prix_selectionnes[0], prix_selectionnes[1])) | (data['Prix Nouméa'].isna())
    filtered_data = data[(data['Taux Remboursement'].isin(taux_selectionnes)) & (price_condition)]

    # --- Corps de la Page ---
    st.title("💊 Med-Explorer")
    search_query = st.text_input("Rechercher un nom de médicament :", placeholder="Ex: DOLIPRANE...")
    st.write("---")

    # Si l'utilisateur a entré une recherche
    if search_query:
        search_results = filtered_data[filtered_data['Nom Complet'].str.contains(search_query, case=False, na=False)]
        
        if not search_results.empty:
            options = ["--- Sélectionnez un médicament pour voir les détails ---"] + search_results['Nom Complet'].tolist()
            selected_med_name = st.selectbox(f"Nous avons trouvé {len(options)-1} médicament(s). Lequel consulter ?", options)

            if selected_med_name != "--- Sélectionnez un médicament pour voir les détails ---":
                final_result = search_results.loc[search_results['Nom Complet'] == selected_med_name].iloc[0]

                # Affichage des détails du médicament sélectionné
                st.subheader(f"Détails pour : **{final_result['Nom Court']}**")
                col1, col2 = st.columns(2)
                prix_noumea = f"{final_result['Prix Nouméa']:,.0f} F CFP".replace(',', ' ') if pd.notna(final_result['Prix Nouméa']) else "Non disponible"
                prix_brousse = f"{final_result['Prix Brousse']:,.0f} F CFP".replace(',', ' ') if pd.notna(final_result['Prix Brousse']) else "Non disponible"
                col1.metric("Prix Nouméa", prix_noumea)
                col2.metric("Prix Brousse", prix_brousse)
                st.info(f"**DCI :** {final_result['DCI']}")
                st.info(f"**Dosage :** {final_result['dosage'] or 'Non spécifié'}")
                st.info(f"**Forme :** {final_result['forme']}")
                
                st.write("---")
                
                # Moteur de similarité hybride
                st.subheader("⚕️ Alternatives Suggérées (Génériques, Autres Marques)")
                
                selected_index = final_result.name
                selected_vector = embeddings[selected_index]
                
                # Étape 1: L'IA trouve les 50 candidats les plus proches
                all_similarities = util.cos_sim(selected_vector, embeddings)[0]
                top_candidates_indices = np.argsort(-all_similarities)[1:51] # Recherche sur 50 candidats
                candidate_df = data.iloc[top_candidates_indices]
                
                # Étape 2: Validation par les règles métier
                validated_alternatives = candidate_df[
                    (candidate_df['DCI'] == final_result['DCI']) &
                    (candidate_df['dosage'] == final_result['dosage']) &
                    (candidate_df['forme'] == final_result['forme']) &
                    (candidate_df['marque'] != final_result['marque'])
                ]
                
                if not validated_alternatives.empty:
                    st.dataframe(
                        validated_alternatives[['Nom Court', 'Prix Nouméa', 'Prix Brousse', 'Taux Remboursement']],
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.write("Aucune alternative directe validée n'a été trouvée parmi les candidats les plus proches.")
        else:
            st.warning("Aucun médicament ne correspond à votre recherche avec les filtres actuels.")
            
    # Si l'utilisateur n'a rien cherché, afficher l'écran d'accueil
    else:
        st.header("Statistiques du jeu de données")
        col1, col2 = st.columns(2)
        col1.metric("Nombre total de médicaments", f"{len(data):,}".replace(',', ' '))
        
        remboursement_counts = data['Taux Remboursement'].value_counts().reset_index()
        remboursement_counts.columns = ['Taux', 'Nombre']
        chart = alt.Chart(remboursement_counts).mark_bar().encode(
            x=alt.X('Taux', sort=None), 
            y='Nombre', 
            tooltip=['Taux', 'Nombre']
        ).properties(title="Répartition par Taux de Remboursement")
        col1.altair_chart(chart, use_container_width=True)
        
        price_chart = alt.Chart(data).mark_bar().encode(
            alt.X("Prix Nouméa:Q", bin=alt.Bin(maxbins=50), title="Prix à Nouméa (F CFP)"), 
            alt.Y('count()', title="Nombre de médicaments"), 
            tooltip=[alt.Tooltip("Prix Nouméa:Q", bin=True), 'count()']
        ).properties(title="Distribution des Prix des Médicaments")
        col2.altair_chart(price_chart, use_container_width=True)