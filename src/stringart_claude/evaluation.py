"""Le juge : noter un rendu indépendamment de la méthode qui l'a produit.

Les solveurs minimisent leur propre objectif, calculé sur leur propre modèle. Les comparer sur
cet objectif reviendrait à les laisser se noter eux-mêmes. Le juge, lui, repart de la séquence
de clous, simule la pièce (`rendu_physique`) et compare cette simulation à l'image cible.

Trois notes, aucune ne suffisant seule :

- `eqm` : écart quadratique moyen brut, pixel à pixel. Sévère avec le grain du fil.
- `eqm_floue` : le même après un léger flou des deux images, comme si l'on reculait de quelques
  pas. C'est la note la plus proche de « est-ce que les tons sont justes ».
- `ssim` : similarité structurelle (1 = identique), sensible aux contours et aux textures plus
  qu'aux niveaux absolus.

Aucune ne remplace le coup d'œil : le notebook montre toujours les images à côté des chiffres.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .rendu import rendu_physique
from .toile import Toile, masque_disque

# Écart-type du flou « vu de loin », en fraction du diamètre : 1 % ≈ 5 mm sur une toile de 50 cm.
FLOU_RELATIF = 0.01


@dataclass(frozen=True)
class Notes:
    eqm: float
    eqm_floue: float
    ssim: float

    def dict(self) -> dict[str, float]:
        return asdict(self)


def comparer(cible: np.ndarray, rendu: np.ndarray) -> Notes:
    """Note un rendu contre sa cible (deux luminances carrées de même taille), dans le disque."""
    from skimage.metrics import structural_similarity

    if cible.shape != rendu.shape:
        raise ValueError(f"formes différentes : cible {cible.shape}, rendu {rendu.shape}")
    cote = cible.shape[0]
    disque = masque_disque(cote)
    a = np.where(disque, cible, 1.0).astype(np.float64)
    b = np.where(disque, rendu, 1.0).astype(np.float64)

    eqm = float(np.sqrt(np.mean((a - b)[disque] ** 2)))
    sigma = FLOU_RELATIF * cote
    ecart_flou = gaussian_filter(a, sigma) - gaussian_filter(b, sigma)
    eqm_floue = float(np.sqrt(np.mean(ecart_flou[disque] ** 2)))
    # Avec full=True, skimage rend (moyenne, carte) ; on moyenne la carte sur le seul disque.
    carte = np.asarray(
        structural_similarity(
            a, b, data_range=1.0, gaussian_weights=True, sigma=1.5,
            use_sample_covariance=False, full=True,
        )[1]  # type: ignore[index]
    )  # fmt: skip
    return Notes(eqm=eqm, eqm_floue=eqm_floue, ssim=float(carte[disque].mean()))


def evaluer(toile: Toile, sequence, cible: np.ndarray) -> Notes:
    """Simule la pièce décrite par `sequence` et la note contre `cible` (`cote × cote`)."""
    return comparer(cible, rendu_physique(toile, sequence, taille=cible.shape[0]))
