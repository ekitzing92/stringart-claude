"""String art calculé : de la photo à la séquence de clous.

Le chemin court :

    from stringart_claude import Toile, creer

    oeuvre = creer("photo.jpg", Toile(diametre_mm=500, n_clous=240))
    oeuvre.enregistrer("sortie/")

Les briques, dans l'ordre où une image les traverse : `image` (cible), `toile` (géométrie),
`moteur` (modèle et objectif), `solveurs`, `sequence` (fil continu), `rendu` et `evaluation`.
`banc` compare les solveurs ; `atelier` et `cli` assemblent le tout.
"""

from .atelier import Oeuvre, creer
from .evaluation import Notes, comparer, evaluer
from .image import charger_image, preparer_cible
from .moteur import Etat, Probleme, image_modele
from .rendu import en_image, planche, rendu_physique, svg
from .sequence import longueur_fil_m, sequencer, texte_sequence
from .solveurs import RECOMMANDE, SOLVEURS, Resultat, resoudre
from .toile import GAMMA, Geometrie, Toile, geometrie

__all__ = [
    "GAMMA",
    "RECOMMANDE",
    "SOLVEURS",
    "Etat",
    "Geometrie",
    "Notes",
    "Oeuvre",
    "Probleme",
    "Resultat",
    "Toile",
    "charger_image",
    "comparer",
    "creer",
    "en_image",
    "evaluer",
    "geometrie",
    "image_modele",
    "longueur_fil_m",
    "planche",
    "preparer_cible",
    "rendu_physique",
    "resoudre",
    "sequencer",
    "svg",
    "texte_sequence",
]
