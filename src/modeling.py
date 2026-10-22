"""Préparation, modèles et fonctions de coût, réutilisés par les notebooks 02 et 03."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "churn_3m"
NUM_COLS = [
    "anciennete_j", "jours_depuis_derniere_co", "releves_7j", "releves_30j",
    "releves_30j_precedents", "jours_actifs_30j", "zones_distinctes_30j",
    "niveau", "gain_niveau_30j", "en_guilde", "nb_guildes", "tendance_activite",
]
CAT_COLS = ["race", "classe"]
# Exclues : char (identifiant) et date_coupure (constante dans chaque jeu)


def make_preprocessor():
    """Imputation + standardisation des numériques, one-hot des catégorielles."""
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])


def rule_14_days(X):
    """Règle métier actuelle : relancer les joueurs absents depuis plus de 14 jours."""
    return (X["jours_depuis_derniere_co"] > 14).astype(int).to_numpy()


# Hypothèses de coût (axe 6)
HYPOTHESES = {
    "abonnement_mois": 12.99,   # € par mois (prix d'un abonnement mensuel WoW en Europe en 2008)
    "mois_gagnes": 12,          # mois d'abonnement supplémentaires d'un joueur retenu
    "cout_offre": 12.99,        # un mois offert = un mois de revenu en moins
    "taux_reussite": 0.25,      # part des joueurs sur le départ que l'offre retient vraiment
}


def campaign_cost(y_true, y_pred, h=HYPOTHESES):
    """Coût total d'une campagne de rétention, en euros.

    FN : valeur du joueur perdue. TP : offre + part des joueurs qui partent quand même.
    FP : offre pour rien. TN : 0.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    valeur = h["abonnement_mois"] * h["mois_gagnes"]
    cout = (fn * valeur
            + tp * (h["cout_offre"] + (1 - h["taux_reussite"]) * valeur)
            + fp * h["cout_offre"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "offres": tp + fp,
            "precision": precision, "recall": recall, "cout": cout}


def threshold_table(y_true, proba, thresholds, h=HYPOTHESES):
    """Precision, Recall et coût total pour plusieurs seuils."""
    rows = []
    for t in thresholds:
        r = campaign_cost(y_true, (proba >= t).astype(int), h)
        rows.append({"seuil": t, **r})
    return pd.DataFrame(rows)


def best_threshold(y_true, proba, h=HYPOTHESES, grid=None):
    """Seuil qui minimise le coût total."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    tab = threshold_table(y_true, proba, grid, h)
    return float(tab.loc[tab["cout"].idxmin(), "seuil"]), tab
