"""Chargement des données WoW et construction des variables par joueur à une date de coupure."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_raw(path=None):
    """Charge wowah_data.csv avec des types compacts et supprime les doublons exacts."""
    path = path or DATA_DIR / "wowah_data.csv"
    df = pd.read_csv(
        path,
        skipinitialspace=True,
        dtype={"char": "int32", "level": "int8", "guild": "int32",
               "race": "category", "charclass": "category", "zone": "category"},
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%y %H:%M:%S")
    return df.drop_duplicates().reset_index(drop=True)


def build_dataset(df, cutoff, horizon_months=3, history_days=90):
    """Une ligne par joueur actif dans les 30 jours avant la coupure.

    Variables : calculées sur les `history_days` jours avant la coupure (fenêtre fixe).
    Cible churn_3m : 1 si le joueur n'apparaît plus dans les `horizon_months` mois suivants.
    """
    cutoff = pd.Timestamp(cutoff)
    past = df[(df["timestamp"] < cutoff)
              & (df["timestamp"] >= cutoff - pd.Timedelta(days=history_days))]
    future = df[(df["timestamp"] >= cutoff)
                & (df["timestamp"] < cutoff + pd.DateOffset(months=horizon_months))]

    last30 = past[past["timestamp"] >= cutoff - pd.Timedelta(days=30)]
    active = last30["char"].unique()
    past = past[past["char"].isin(active)]
    last30 = last30[last30["char"].isin(active)]
    prev30 = past[(past["timestamp"] >= cutoff - pd.Timedelta(days=60))
                  & (past["timestamp"] < cutoff - pd.Timedelta(days=30))]
    last7 = last30[last30["timestamp"] >= cutoff - pd.Timedelta(days=7)]

    past = past.sort_values("timestamp")
    g = past.groupby("char")
    feat = pd.DataFrame(index=pd.Index(active, name="char"))
    # Ancienneté et récence (en jours)
    feat["anciennete_j"] = (cutoff - g["timestamp"].min()).dt.days
    feat["jours_depuis_derniere_co"] = (cutoff - g["timestamp"].max()).dt.total_seconds() / 86400
    # Activité (1 relevé ≈ 10 min de jeu)
    feat["releves_7j"] = last7.groupby("char").size()
    feat["releves_30j"] = last30.groupby("char").size()
    feat["releves_30j_precedents"] = prev30.groupby("char").size()
    feat["jours_actifs_30j"] = last30.groupby("char")["timestamp"].apply(lambda s: s.dt.normalize().nunique())
    feat["zones_distinctes_30j"] = last30.groupby("char")["zone"].nunique()
    # Progression
    feat["niveau"] = g["level"].last()
    feat["gain_niveau_30j"] = last30.groupby("char")["level"].max() - last30.groupby("char")["level"].min()
    # Guilde
    feat["en_guilde"] = (g["guild"].last() != -1).astype(int)
    feat["nb_guildes"] = past[past["guild"] != -1].groupby("char")["guild"].nunique()
    # Profil
    feat["race"] = g["race"].last().astype(str)
    feat["classe"] = g["charclass"].last().astype(str)

    feat = feat.fillna({"releves_7j": 0, "releves_30j_precedents": 0, "gain_niveau_30j": 0, "nb_guildes": 0})
    # Tendance d'activité : mois récent / mois précédent
    feat["tendance_activite"] = feat["releves_30j"] / (feat["releves_30j_precedents"] + 1)

    returning = set(future["char"].unique())
    feat["churn_3m"] = (~feat.index.isin(returning)).astype(int)
    feat["date_coupure"] = cutoff
    return feat.reset_index()
