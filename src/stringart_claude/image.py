"""De la photo à la cible : chargement, cadrage dans le disque, mise en tons.

Tout le projet représente une image par un tableau 2D de flottants dans [0, 1], en valeurs
*perçues* (celles d'un fichier image ordinaire) : 1 est le blanc de la toile nue, 0 le noir.
Cible, prédiction du modèle et simulation de la pièce partagent cette convention et se
comparent donc directement.

La mise en tons n'est pas cosmétique : c'est le premier levier sur la qualité du résultat (voir
le notebook). Une toile de fils ne sait pas tout faire — pas de noir profond à côté d'un blanc
pur, pas de nuance plus fine que quelques fils — et la cible doit lui demander ce qu'elle sait
rendre : du contraste local partout, et des noirs qui restent à sa portée.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .toile import masque_disque

CONTRASTES = ("aucun", "auto", "local")


def charger_image(chemin: str | Path) -> np.ndarray:
    """Lit un fichier image et rend sa luminance. La transparence est composée sur du blanc."""
    with Image.open(chemin) as fichier:
        rgba = fichier.convert("RGBA")
    fond = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    fond.alpha_composite(rgba)
    return np.asarray(fond.convert("L"), dtype=np.float32) / 255.0


def preparer_cible(
    image: np.ndarray,
    cote: int,
    *,
    contraste: str = "local",
    relief: float = 0.02,
    gamma: float = 1.0,
    noir: float = 0.3,
    zoom: float = 1.0,
    decalage: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Cadre l'image dans le disque de la toile et ajuste ses tons.

    - `zoom` > 1 resserre le cadrage autour du centre ; `decalage = (dx, dy)` déplace ce centre,
      en fraction du petit côté de l'image (dx > 0 vers la droite, dy > 0 vers le bas).
    - `contraste` : `"aucun"` laisse les tons tels quels ; `"auto"` étire l'histogramme pour
      occuper toute la plage du blanc au noir ; `"local"` ajoute une égalisation adaptative
      (CLAHE), qui fait ressortir le relief dans les zones uniformément claires ou sombres.
      C'est le défaut : sur une photo, c'est ce qui fait apparaître les traits d'un visage.
    - `relief` : force du contraste local, de 0 (aucun effet) à 0,05 environ. Il n'y a pas de
      valeur universelle : un visage lisse en demande beaucoup (0,03), une photo déjà chargée
      de textures — pelage, feuillage — très peu (0,01), sans quoi tout devient détail.
    - `gamma` > 1 assombrit les tons moyens (plus de fil), < 1 les éclaircit.
    - `noir` : valeur demandée pour le noir de l'image, entre 0 et 1. Les tons sont comprimés
      dans `[noir, 1]`. Un noir profond coûte des dizaines de passages par pixel, qui voilent
      tout ce qu'ils traversent ; viser un gris foncé donne une image plus nette pour moins de
      fil. `noir=0` désactive la compression.

    Les pixels hors du disque sont mis à blanc. Rend un tableau `cote × cote` en `float32`.
    """
    if contraste not in CONTRASTES:
        raise ValueError(f"contraste inconnu : {contraste!r} (choix : {', '.join(CONTRASTES)})")
    if image.ndim != 2:
        raise ValueError("image en luminance attendue (tableau 2D) : passer par charger_image")
    if zoom <= 0 or gamma <= 0:
        raise ValueError("zoom et gamma doivent être strictement positifs")
    if not 0.0 <= noir < 1.0:
        raise ValueError("noir doit être dans [0, 1[")
    if not 0.0 <= relief <= 1.0:
        raise ValueError("relief doit être dans [0, 1]")

    hauteur, largeur = image.shape
    petit = min(hauteur, largeur)
    taille = petit / zoom
    cx = largeur / 2.0 + decalage[0] * petit
    cy = hauteur / 2.0 + decalage[1] * petit
    boite = (cx - taille / 2, cy - taille / 2, cx + taille / 2, cy + taille / 2)
    # Mode "F" : flottants 32 bits. Ce qui sort du cadre d'origine est complété en blanc.
    source = Image.fromarray(np.asarray(image, dtype=np.float32), mode="F")
    cadree = source.transform(
        (cote, cote),
        Image.Transform.EXTENT,
        boite,
        resample=Image.Resampling.BICUBIC,
        fillcolor=1.0,
    )
    # EXTENT ne filtre pas avant de réduire : on passe par une taille intermédiaire entière
    # puis une réduction en moyenne, pour ne pas créer de moiré sur les photos très détaillées.
    facteur = max(1, int(taille // cote))
    if facteur > 1:
        grande = source.transform(
            (cote * facteur, cote * facteur),
            Image.Transform.EXTENT,
            boite,
            resample=Image.Resampling.BICUBIC,
            fillcolor=1.0,
        )
        cadree = grande.reduce(facteur)
    luminance = np.clip(np.asarray(cadree, dtype=np.float32), 0.0, 1.0)

    disque = masque_disque(cote)
    if contraste in ("auto", "local"):
        bas, haut = np.percentile(luminance[disque], [1.0, 99.0])
        if haut - bas > 1e-3:
            luminance = np.clip((luminance - bas) / (haut - bas), 0.0, 1.0)
    if contraste == "local" and relief > 0:
        from skimage.exposure import equalize_adapthist

        luminance = equalize_adapthist(luminance, clip_limit=relief).astype(np.float32)
    if gamma != 1.0:
        luminance = luminance**gamma
    luminance = noir + (1.0 - noir) * luminance

    return np.where(disque, luminance, 1.0).astype(np.float32)
