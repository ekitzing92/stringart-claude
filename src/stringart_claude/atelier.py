"""L'atelier : la chaîne complète, d'un fichier image aux livrables.

C'est le point d'entrée de la partie « application ». `creer` enchaîne la préparation de
l'image, la résolution et la simulation ; l'`Oeuvre` qu'elle rend sait s'enregistrer dans un
dossier, sous toutes les formes utiles pour passer à la réalisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evaluation import Notes, comparer
from .guide import guide_html
from .image import charger_image, preparer_cible
from .moteur import Probleme
from .rendu import en_image, planche, rendu_physique, svg
from .sequence import est_pas_de_bord, longueur_fil_m, texte_sequence
from .solveurs import RECOMMANDE, Resultat, Suivi, resoudre
from .toile import Toile, geometrie


@dataclass
class Oeuvre:
    """Une pièce calculée : sa toile, sa cible, sa solution, et la simulation du résultat."""

    toile: Toile
    cible: np.ndarray
    resultat: Resultat
    rendu: np.ndarray  # simulation de la pièce, à la résolution de travail
    notes: Notes

    @property
    def sequence(self) -> np.ndarray:
        return self.resultat.sequence

    @property
    def fil_m(self) -> float:
        return longueur_fil_m(self.toile, self.sequence)

    def apercu(self, taille: int = 1000) -> np.ndarray:
        """La simulation de la pièce, à la taille demandée."""
        return rendu_physique(self.toile, self.sequence, taille=taille)

    def resume(self) -> str:
        """Quelques lignes : ce qu'il faut savoir avant de sortir le marteau."""
        n_bord = int(est_pas_de_bord(self.sequence, self.toile.n_clous).sum())
        # Un quart de minute par passage : ordre de grandeur d'un montage à la main, une fois
        # le geste pris.
        heures = (len(self.sequence) - 1) * 15 / 3600
        return "\n".join(
            [
                f"Toile de {self.toile.diametre_mm:g} mm, {self.toile.n_clous} clous, "
                f"fil de {self.toile.epaisseur_fil_mm:g} mm.",
                f"Méthode : {self.resultat.nom} ({self.resultat.duree_s:.1f} s de calcul, "
                f"arrêt : {self.resultat.arret}).",
                f"{self.resultat.n_cordes} cordes, {len(self.sequence) - 1} passages "
                f"dont {n_bord} le long du bord.",
                f"Environ {self.fil_m:.0f} m de fil et {heures:.0f} h de montage "
                "(à 15 s le passage).",
                f"Fidélité simulée : EQM floue {self.notes.eqm_floue:.3f}, "
                f"SSIM {self.notes.ssim:.2f}.",
            ]
        )

    def enregistrer(self, dossier: str | Path, taille: int = 1000) -> dict[str, Path]:
        """Écrit les livrables dans `dossier` (créé au besoin) et rend leurs chemins.

        - `rendu.png` : la simulation de la pièce ;
        - `comparaison.png` : la cible à côté de la simulation ;
        - `sequence.txt` : la suite de clous, à imprimer ;
        - `guide.html` : la même, pas à pas, à ouvrir sur un téléphone posé près de la toile ;
        - `piece.svg` : le tracé à l'échelle 1, gabarit de perçage compris.
        """
        dossier = Path(dossier)
        dossier.mkdir(parents=True, exist_ok=True)
        apercu = self.apercu(taille)
        chemins = {
            "rendu": dossier / "rendu.png",
            "comparaison": dossier / "comparaison.png",
            "sequence": dossier / "sequence.txt",
            "guide": dossier / "guide.html",
            "svg": dossier / "piece.svg",
        }
        en_image(apercu).save(chemins["rendu"])
        planche([self.cible, apercu], taille=min(taille, 600)).save(chemins["comparaison"])
        chemins["sequence"].write_text(texte_sequence(self.toile, self.sequence), "utf-8")
        chemins["guide"].write_text(guide_html(self.toile, self.sequence), "utf-8")
        chemins["svg"].write_text(svg(self.toile, self.sequence), "utf-8")
        return chemins


def creer(
    image: str | Path | np.ndarray,
    toile: Toile | None = None,
    *,
    solveur: str = RECOMMANDE,
    contraste: str = "local",
    relief: float = 0.02,
    gamma: float = 1.0,
    noir: float = 0.3,
    zoom: float = 1.0,
    decalage: tuple[float, float] = (0.0, 0.0),
    cordes_max: int | None = None,
    suivi: Suivi | None = None,
    **options_solveur,
) -> Oeuvre:
    """Calcule une pièce à partir d'une image (chemin de fichier ou tableau déjà chargé).

    Les réglages d'image (`contraste`, `relief`, `gamma`, `noir`, `zoom`, `decalage`) sont ceux de
    `preparer_cible`. `cordes_max` plafonne le nombre de cordes, donc le temps de montage.
    `suivi(etat, n_coups)` est appelé régulièrement pendant la résolution.
    """
    toile = toile or Toile()
    brute = image if isinstance(image, np.ndarray) else charger_image(image)
    cible = preparer_cible(
        brute, toile.cote, contraste=contraste, relief=relief, gamma=gamma, noir=noir,
        zoom=zoom, decalage=decalage,
    )  # fmt: skip
    probleme = Probleme(geometrie(toile), cible)
    resultat = resoudre(probleme, solveur, cordes_max=cordes_max, suivi=suivi, **options_solveur)
    rendu = rendu_physique(toile, resultat.sequence)
    return Oeuvre(toile, cible, resultat, rendu, comparer(cible, rendu))
