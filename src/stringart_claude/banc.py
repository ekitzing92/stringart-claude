"""Le banc d'essai : faire tourner plusieurs méthodes sur les mêmes problèmes et les comparer.

Chaque passage produit une `Mesure` : l'objectif atteint (ce que le solveur optimise), les notes
du juge (ce que donnerait la pièce, voir `evaluation.py`), et ce que la solution coûte à
calculer et à monter. Le classement agrège les mesures par méthode.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .evaluation import evaluer
from .moteur import Probleme
from .sequence import est_pas_de_bord, longueur_fil_m
from .solveurs import SOLVEURS, Resultat, resoudre


@dataclass(frozen=True)
class Mesure:
    """Un solveur, un problème, les chiffres."""

    probleme: str
    solveur: str
    ecart_modele: float  # racine de l'objectif : écart quadratique moyen vu par le solveur
    eqm: float  # les trois notes du juge, sur la pièce simulée
    eqm_floue: float
    ssim: float
    cordes: int
    pas_de_bord: int
    fil_m: float
    duree_s: float
    arret: str


def mesurer(nom_probleme: str, probleme: Probleme, resultat: Resultat) -> Mesure:
    """Note un résultat : objectif, juge, coût de montage."""
    notes = evaluer(probleme.toile, resultat.sequence, probleme.cible)
    return Mesure(
        probleme=nom_probleme,
        solveur=resultat.nom,
        ecart_modele=float(np.sqrt(max(resultat.cout, 0.0))),
        eqm=notes.eqm,
        eqm_floue=notes.eqm_floue,
        ssim=notes.ssim,
        cordes=resultat.n_cordes,
        pas_de_bord=int(est_pas_de_bord(resultat.sequence, probleme.toile.n_clous).sum()),
        fil_m=longueur_fil_m(probleme.toile, resultat.sequence),
        duree_s=resultat.duree_s,
        arret=resultat.arret,
    )


def lancer(
    problemes: dict[str, Probleme],
    solveurs: Iterable[str] | None = None,
    options: dict[str, dict] | None = None,
    garder_resultats: bool = False,
) -> tuple[list[Mesure], dict[tuple[str, str], Resultat]]:
    """Fait tourner chaque solveur sur chaque problème.

    `options` associe à un nom de solveur les arguments à lui passer. Rend les mesures et, si
    `garder_resultats`, les résultats complets indexés par `(problème, solveur)`.
    """
    noms = list(SOLVEURS) if solveurs is None else list(solveurs)
    options = options or {}
    mesures: list[Mesure] = []
    resultats: dict[tuple[str, str], Resultat] = {}
    for nom_probleme, probleme in problemes.items():
        for nom in noms:
            resultat = resoudre(probleme, nom, **options.get(nom, {}))
            mesures.append(mesurer(nom_probleme, probleme, resultat))
            if garder_resultats:
                resultats[(nom_probleme, nom)] = resultat
    return mesures, resultats


def classement(mesures: list[Mesure]) -> list[dict]:
    """Agrège par solveur, du meilleur au moins bon.

    Pour chaque problème, l'objectif de chaque solveur est rapporté au meilleur objectif obtenu
    sur ce problème ; `retard_pct` est la moyenne de ces retards. Les autres colonnes sont des
    moyennes simples.
    """
    meilleur: dict[str, float] = {}
    for m in mesures:
        meilleur[m.probleme] = min(meilleur.get(m.probleme, np.inf), m.ecart_modele**2)
    lignes = []
    for nom in dict.fromkeys(m.solveur for m in mesures):
        siennes = [m for m in mesures if m.solveur == nom]
        lignes.append(
            {
                "solveur": nom,
                "retard_pct": float(
                    np.mean([100 * (m.ecart_modele**2 / meilleur[m.probleme] - 1) for m in siennes])
                ),
                "eqm_floue": float(np.mean([m.eqm_floue for m in siennes])),
                "ssim": float(np.mean([m.ssim for m in siennes])),
                "cordes": float(np.mean([m.cordes for m in siennes])),
                "duree_s": float(np.mean([m.duree_s for m in siennes])),
            }
        )
    return sorted(lignes, key=lambda ligne: ligne["retard_pct"])


def tableau_mesures(mesures: list[Mesure]) -> str:
    """Les mesures en tableau Markdown, une ligne par passage."""
    lignes = [
        "| problème | solveur | écart modèle | EQM | EQM floue | SSIM | cordes | fil (m) "
        "| durée (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in mesures:
        lignes.append(
            f"| {m.probleme} | {m.solveur} | {m.ecart_modele:.4f} | {m.eqm:.4f} | "
            f"{m.eqm_floue:.4f} | {m.ssim:.3f} | {m.cordes} | {m.fil_m:.0f} | {m.duree_s:.1f} |"
        )
    return "\n".join(lignes)


def tableau_classement(mesures: list[Mesure]) -> str:
    """Le classement en tableau Markdown."""
    lignes = [
        "| solveur | retard sur le meilleur objectif | EQM floue | SSIM | cordes | durée (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ligne in classement(mesures):
        lignes.append(
            f"| {ligne['solveur']} | +{ligne['retard_pct']:.2f} % | {ligne['eqm_floue']:.4f} | "
            f"{ligne['ssim']:.3f} | {ligne['cordes']:.0f} | {ligne['duree_s']:.1f} |"
        )
    return "\n".join(lignes)
