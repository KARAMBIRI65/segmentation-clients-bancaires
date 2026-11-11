"""Segmentation de clients bancaires par K-means.

Pipeline complet : chargement -> variables -> choix de k -> K-means
-> profils des segments -> recommandations business -> figures.

Lancement :  python -m src.segmentation   (depuis la racine du projet)
"""
from pathlib import Path

import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RACINE = Path(__file__).resolve().parents[1]
FICHIER_DONNEES = RACINE / "data" / "bank-additional-full.csv"
DOSSIER_FIGURES = RACINE / "figures"
DOSSIER_RESULTATS = RACINE / "resultats"

GRAINE = 42
K_RETENU = 5

# Palette catégorielle (ordre fixe, validée daltonisme) et style sobre
COULEURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GRIS = "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b5b4ae", "axes.labelcolor": GRIS,
    "xtick.color": GRIS, "ytick.color": GRIS,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

NIVEAU_ETUDES = {
    "illiterate": 0, "basic.4y": 1, "basic.6y": 2, "basic.9y": 3,
    "high.school": 4, "professional.course": 5, "university.degree": 6,
}
METIERS_QUALIFIES = {"admin.", "management", "technician", "entrepreneur", "self-employed"}

LIBELLES = {
    "age": "Âge",
    "niveau_etudes": "Niveau d'études (0-6)",
    "emploi_qualifie": "Emploi qualifié",
    "retraite": "Retraité",
    "etudiant": "Étudiant",
    "marie": "Marié(e)",
    "credit_immo": "Crédit immobilier",
    "credit_conso": "Crédit conso",
    "defaut_inconnu": "Historique crédit inconnu",
    "deja_contacte": "Déjà contacté avant",
}


# ---------------------------------------------------------------- données
def charger_donnees(chemin=FICHIER_DONNEES):
    """Charge le jeu Bank Marketing (UCI, 41 188 clients)."""
    return pd.read_csv(chemin, sep=";")


def construire_variables(df):
    """Transforme les colonnes brutes en variables numériques décrivant le CLIENT.

    On garde uniquement des informations sur la personne (profil, crédits,
    relation passée avec la banque). On exclut volontairement :
      - `duration` (durée de l'appel) : connue seulement après l'appel ;
      - les indicateurs macro-économiques : ils décrivent le contexte, pas le client ;
      - `y` (souscription) : on la garde de côté pour ÉVALUER les segments.
    """
    X = pd.DataFrame(index=df.index)
    X["age"] = df["age"]
    X["niveau_etudes"] = df["education"].map(NIVEAU_ETUDES)
    X["niveau_etudes"] = X["niveau_etudes"].fillna(X["niveau_etudes"].median())
    X["emploi_qualifie"] = df["job"].isin(METIERS_QUALIFIES).astype(int)
    X["retraite"] = (df["job"] == "retired").astype(int)
    X["etudiant"] = (df["job"] == "student").astype(int)
    X["marie"] = (df["marital"] == "married").astype(int)
    X["credit_immo"] = (df["housing"] == "yes").astype(int)
    X["credit_conso"] = (df["loan"] == "yes").astype(int)
    X["defaut_inconnu"] = (df["default"] == "unknown").astype(int)
    X["deja_contacte"] = (df["previous"] > 0).astype(int)
    return X


def standardiser(X):
    """Centre-réduit les variables (K-means est sensible aux échelles)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


# ---------------------------------------------------------------- choix de k
def evaluer_k(Z, k_min=2, k_max=10, taille_echantillon=8000):
    """Inertie (méthode du coude) et score de silhouette pour chaque k."""
    rng = np.random.RandomState(GRAINE)
    idx = rng.choice(len(Z), size=min(taille_echantillon, len(Z)), replace=False)
    lignes = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=GRAINE).fit(Z)
        sil = silhouette_score(Z[idx], km.labels_[idx])
        lignes.append({"k": k, "inertie": km.inertia_, "silhouette": sil})
    return pd.DataFrame(lignes)


def entrainer_kmeans(Z, k=K_RETENU):
    return KMeans(n_clusters=k, n_init=20, random_state=GRAINE).fit(Z)


# ---------------------------------------------------------------- profils
def nommer_segment(profil):
    """Donne un nom lisible à un segment à partir de son profil moyen."""
    if profil["etudiant"] > 0.5:
        return "Jeunes étudiants"
    if profil["retraite"] > 0.5:
        return "Seniors retraités"
    if profil["defaut_inconnu"] > 0.5:
        return "Clients peu documentés"
    if profil["emploi_qualifie"] > 0.5:
        return "Actifs qualifiés"
    return "Actifs employés / ouvriers"


def profiler_segments(df, X, labels):
    """Moyennes par segment + taille + taux de souscription (y)."""
    data = X.copy()
    data["segment"] = labels
    data["souscription"] = (df["y"] == "yes").astype(int)
    profils = data.groupby("segment").mean()
    profils["nb_clients"] = data.groupby("segment").size()
    profils["part_clients"] = profils["nb_clients"] / len(data)
    profils["nom"] = profils.apply(nommer_segment, axis=1)
    # Lift : combien de fois plus (ou moins) que la moyenne le segment souscrit
    profils["lift"] = profils["souscription"] / data["souscription"].mean()
    return profils.sort_values("souscription", ascending=False)


RECOMMANDATIONS = {
    "Jeunes étudiants": (
        "Taux de souscription le plus élevé. Proposer un livret épargne jeune "
        "avec versements programmés, via l'app mobile et les partenariats écoles. "
        "Enjeu long terme : fidéliser avant le premier emploi."
    ),
    "Seniors retraités": (
        "Forte appétence pour l'épargne sécurisée. Conseiller dédié, dépôt à terme, "
        "assurance-vie et transmission. Contact téléphonique ou agence."
    ),
    "Actifs qualifiés": (
        "Plus gros segment, taux proche de la moyenne. Offres digitales "
        "(épargne en ligne, PEA, packs premium) et ciblage fin par scoring "
        "plutôt qu'appels de masse."
    ),
    "Actifs employés / ouvriers": (
        "Souscription sous la moyenne. Produits simples à petits montants "
        "(épargne automatique, arrondi des achats) et éducation financière ; "
        "limiter les relances téléphoniques coûteuses."
    ),
    "Clients peu documentés": (
        "Taux le plus faible. Priorité : compléter la connaissance client (KYC, "
        "historique de crédit) avant toute campagne commerciale. Exclure des "
        "campagnes d'appels pour réduire les coûts."
    ),
}


def tableau_recommandations(profils):
    t = profils[["nom", "nb_clients", "part_clients", "age", "souscription", "lift"]].copy()
    t["recommandation"] = t["nom"].map(RECOMMANDATIONS)
    return t


# ---------------------------------------------------------------- figures
def _sauver(fig, nom):
    DOSSIER_FIGURES.mkdir(exist_ok=True)
    fig.savefig(DOSSIER_FIGURES / nom, bbox_inches="tight", facecolor="white")


def figure_choix_k(scores, k_retenu=K_RETENU):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, titre in [
        (axes[0], "inertie", "Méthode du coude (inertie)"),
        (axes[1], "silhouette", "Score de silhouette"),
    ]:
        ax.plot(scores["k"], scores[col], color=COULEURS[0], lw=2, marker="o", ms=6)
        ligne = scores[scores["k"] == k_retenu].iloc[0]
        ax.plot(k_retenu, ligne[col], marker="o", ms=10, color=COULEURS[1], zorder=3)
        ax.annotate(f"k = {k_retenu}", (k_retenu, ligne[col]), xytext=(8, 8),
                    textcoords="offset points", color=GRIS)
        ax.set_title(titre, loc="left", fontsize=11)
        ax.set_xlabel("Nombre de segments k")
        ax.set_xticks(scores["k"])
    fig.tight_layout()
    _sauver(fig, "01_choix_k.png")
    return fig


def _couleur_par_nom(profils):
    noms = sorted(profils["nom"].unique())
    return {n: COULEURS[i % len(COULEURS)] for i, n in enumerate(noms)}


def figure_pca(Z, labels, profils, n_points=6000):
    """Projection 2D (ACP) des clients colorés par segment."""
    pca = PCA(n_components=2, random_state=GRAINE)
    P = pca.fit_transform(Z)
    rng = np.random.RandomState(GRAINE)
    idx = rng.choice(len(P), size=min(n_points, len(P)), replace=False)
    couleurs = _couleur_par_nom(profils)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for seg, ligne in profils.iterrows():
        m = labels[idx] == seg
        ax.scatter(P[idx][m, 0], P[idx][m, 1], s=9, alpha=0.55,
                   color=couleurs[ligne["nom"]], label=ligne["nom"],
                   edgecolors="none")
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"Composante 1 ({var[0]:.0%} de la variance)")
    ax.set_ylabel(f"Composante 2 ({var[1]:.0%} de la variance)")
    ax.set_title("Les clients projetés en 2D, colorés par segment", loc="left", fontsize=11)
    ax.legend(frameon=False, markerscale=2.2, loc="best", fontsize=9)
    fig.tight_layout()
    _sauver(fig, "02_projection_acp.png")
    return fig


def figure_profils(X, profils):
    """Heatmap : écart de chaque segment à la moyenne (en écarts-types)."""
    cols = list(LIBELLES)
    ecarts = (profils[cols] - X[cols].mean()) / X[cols].std()
    ecarts.index = profils["nom"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    lim = 2.0
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    im = ax.imshow(ecarts.clip(-lim, lim).values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([LIBELLES[c] for c in cols], rotation=35, ha="right")
    ax.set_yticks(range(len(ecarts)))
    ax.set_yticklabels(ecarts.index)
    ax.grid(False)
    for i in range(ecarts.shape[0]):
        for j in range(ecarts.shape[1]):
            v = profils[cols[j]].iloc[i]
            txt = f"{v:.0f}" if cols[j] == "age" else (f"{v:.1f}" if cols[j] == "niveau_etudes" else f"{v:.0%}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color="#0b0b0b")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Écart à la moyenne (écarts-types)", color=GRIS)
    ax.set_title("Profil de chaque segment (valeur moyenne, couleur = écart à la moyenne)",
                 loc="left", fontsize=11)
    fig.tight_layout()
    _sauver(fig, "03_profils_segments.png")
    return fig


def figure_souscription(profils, taux_global):
    """Taux de souscription au dépôt à terme par segment."""
    couleurs = _couleur_par_nom(profils)
    p = profils.sort_values("souscription")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(p["nom"], p["souscription"], color=[couleurs[n] for n in p["nom"]], height=0.6)
    ax.axvline(taux_global, color=GRIS, lw=1.2, ls="--")
    ax.text(taux_global, -0.75, f" moyenne {taux_global:.1%}", color=GRIS, fontsize=9, va="bottom")
    for i, (v, n) in enumerate(zip(p["souscription"], p["nb_clients"])):
        ax.text(v + 0.004, i, f"{v:.1%}  ({n:,} clients)".replace(",", " "),
                va="center", fontsize=9, color="#0b0b0b",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_xlim(0, p["souscription"].max() * 1.45)
    ax.set_ylim(-0.85, len(p) - 0.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_title("Taux de souscription au dépôt à terme par segment", loc="left", fontsize=11)
    fig.tight_layout()
    _sauver(fig, "04_souscription_par_segment.png")
    return fig


# ---------------------------------------------------------------- pipeline
def main():
    plt.switch_backend("Agg")  # pas de fenêtre graphique en ligne de commande
    df = charger_donnees()
    X = construire_variables(df)
    Z, _ = standardiser(X)

    scores = evaluer_k(Z)
    print("Choix de k :\n", scores.round(3).to_string(index=False))

    km = entrainer_kmeans(Z, K_RETENU)
    profils = profiler_segments(df, X, km.labels_)
    reco = tableau_recommandations(profils)

    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    profils.round(3).to_csv(DOSSIER_RESULTATS / "profils_segments.csv")
    reco.round(3).to_csv(DOSSIER_RESULTATS / "recommandations.csv", index=False)
    pd.DataFrame({"segment": km.labels_}).assign(
        nom=lambda d: d["segment"].map(profils["nom"])
    ).to_csv(DOSSIER_RESULTATS / "segment_par_client.csv", index_label="client_id")

    taux_global = (df["y"] == "yes").mean()
    figure_choix_k(scores)
    figure_pca(Z, km.labels_, profils)
    figure_profils(X, profils)
    figure_souscription(profils, taux_global)

    print("\nSegments :")
    print(reco[["nom", "nb_clients", "souscription", "lift"]].round(3).to_string())
    print(f"\nTaux de souscription global : {taux_global:.1%}")


if __name__ == "__main__":
    main()
