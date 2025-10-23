import spacy
from spacy.matcher import Matcher
import re
import pandas as pd

FORM_MAP = {
    'cpr': 'Comprimé', 'comprimé': 'Comprimé', 'cp': 'Comprimé',
    'gél': 'Gélule', 'gélule': 'Gélule',
    'syr': 'Sirop', 'sirop': 'Sirop',
    'sol': 'Solution', 'solution': 'Solution', 's': 'Solution',
    'susp': 'Suspension', 'suspension': 'Suspension',
    'inj': 'Injectable', 'injectable': 'Injectable',
    'perf': 'Perfusion', 'perfusion': 'Perfusion',
    'pom': 'Pommade', 'pommade': 'Pommade',
    'cr': 'Crème', 'crème': 'Crème',
    'pulv': 'Pulvérisation', 'pulvérisation': 'Pulvérisation',
    'collyre': 'Collyre', 'col': 'Collyre',
    'gte': 'Gouttes', 'goutte': 'Gouttes', 'gt': 'Gouttes'
}

FORM_QUALIFIERS_MAP = {
    'lp': 'LP', 'eff': 'Effervescent', 'orodisp': 'Orodispersible',
    'séc': 'Sécable', 'gast': 'Gastro-résistant', 'buv': 'Buvable',
    'opht': 'Ophtalmique', 'auric': 'Auriculaire'
}

class FeatureExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
        except OSError:
            print("Modèle 'fr_core_news_sm' non trouvé.")
            from spacy.lang.fr import French
            self.nlp = French()
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()

    def _setup_patterns(self):
        units = ['mg', 'g', '%', 'µg', 'ui', 'ch', 'dh', 'k', 'mk']
        pattern_simple = [{"LIKE_NUM": True}, {"LOWER": {"IN": units}}]
        self.matcher.add("DOSAGE", [pattern_simple])
        pattern_double = [{"LIKE_NUM": True}, {"LOWER": "mg"}, {"TEXT": "/"}, {"LIKE_NUM": True}, {"LOWER": "mg"}]
        self.matcher.add("DOSAGE", [pattern_double])
        pattern_concentration = [{"LIKE_NUM": True}, {"LOWER": {"IN": units}}, {"TEXT": "/"}, {"LIKE_NUM": True, "OP": "?"}, {"LOWER": "ml"}]
        self.matcher.add("DOSAGE", [pattern_concentration])
        pattern_action = [{"LIKE_NUM": True}, {"LOWER": {"IN": units}}, {"TEXT": "/"}, {"LOWER": {"IN": ["pulverisation", "pulv", "dose", "inhalation"]}}]
        self.matcher.add("DOSAGE", [pattern_action])

    def _process_doc(self, doc, original_libelle):
        matches = self.matcher(doc)
        best_dosage_match = max(matches, key=lambda m: m[2] - m[1], default=None)
        dosage_text = doc[best_dosage_match[1]:best_dosage_match[2]].text.replace(' ', '') if best_dosage_match else None
        
        first_dosage_start_char_index = doc[best_dosage_match[1]].idx if best_dosage_match else -1
        marque = original_libelle[:first_dosage_start_char_index].strip() if first_dosage_start_char_index != -1 else None
        if not marque:
            match = re.search(r'^([^\d]+)', original_libelle)
            marque = match.group(1).strip() if match else original_libelle

        formes_trouvees = {FORM_MAP[token.lower_] for token in doc if token.lower_ in FORM_MAP}
        qualifiers_trouves = {FORM_QUALIFIERS_MAP[token.lower_] for token in doc if token.lower_ in FORM_QUALIFIERS_MAP}
        
        forme_base = " ".join(sorted(list(formes_trouvees))) if formes_trouvees else "Non spécifiée"
        forme_finale = f"{forme_base} {' '.join(sorted(list(qualifiers_trouves)))}".strip() if qualifiers_trouves else forme_base

        return {"marque": marque, "dosage": dosage_text, "forme": forme_finale}

    def extract_batch(self, libelles):
        original_libelles = [str(l) if pd.notna(l) else "" for l in libelles]
        
        normalized_libelles = []
        for l in original_libelles:
            temp_l = l.replace('/', ' / ')
            temp_l = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', temp_l)
            normalized_libelles.append(temp_l)
            
        docs = self.nlp.pipe(normalized_libelles)
        
        return [self._process_doc(doc, original_libelles[i]) for i, doc in enumerate(docs)]

feature_extractor_instance = FeatureExtractor()