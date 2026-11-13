"""Les rendus : la simulation physique de la pièce, les planches de comparaison, le SVG.

`rendu_physique` ne passe **pas** par le modèle des solveurs. Il dessine chaque fil comme un
trait noir opaque à sa vraie largeur, sur une image assez grande pour que le fil y fasse
plusieurs pixels, puis réduit par moyenne. C'est une seconde implémentation, indépendante et
plus proche de la réalité, de ce que donnera la toile : c'est elle qu'on montre et elle qui
sert de juge (`evaluation.py`). Si le modèle des solveurs était faux, c'est ici qu'on le verrait.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from PIL import Image

from .toile import GAMMA, Toile, positions_clous

# Largeur visée du fil, en pixels, dans l'image de dessin (avant réduction).
_PIXELS_PAR_FIL = 3.0
# Au-delà, l'image de dessin devient déraisonnable : on borne (le fil y fait moins de pixels).
_COTE_DESSIN_MAX = 9000


@njit(cache=True)
def _tracer_fils(dessin, points, demi_largeur):
    """Dessine les fils dans `dessin`, où chaque pixel porte sa part visible, de 0 à 255.

    `points` : sommets `(ligne, colonne)` de la ligne brisée, en coordonnées continues. On
    avance d'un pixel le long de l'axe principal du fil ; à chaque pas, le fil occupe sur l'autre
    axe un intervalle dont on reporte la longueur exacte dans les pixels touchés — entiers au
    cœur du fil, fractionnaires sur ses bords. La surface cachée est donc juste quelles que
    soient la pente, la position et la largeur du fil. Deux fils qui se chevauchent dans un
    même pixel ne cachent que le plus couvrant des deux (un fil est opaque).
    """
    cote = dessin.shape[0]
    for s in range(points.shape[0] - 1):
        r0, c0 = points[s, 0], points[s, 1]
        r1, c1 = points[s + 1, 0], points[s + 1, 1]
        if r0 == r1 and c0 == c1:
            continue
        horizontal = abs(c1 - c0) >= abs(r1 - r0)
        # (u, v) : u le long de l'axe principal, v sur l'axe secondaire.
        if horizontal:
            u0, v0, u1, v1 = c0, r0, c1, r1
        else:
            u0, v0, u1, v1 = r0, c0, r1, c1
        if u0 > u1:
            u0, v0, u1, v1 = u1, v1, u0, v0
        pente = (v1 - v0) / (u1 - u0)
        demi = demi_largeur * np.sqrt(1.0 + pente * pente)
        for a in range(max(0, int(np.floor(u0))), min(cote, int(np.ceil(u1)))):
            # Part de ce pas réellement parcourue par le fil (les deux bouts sont partiels).
            long_pas = min(u1, a + 1.0) - max(u0, float(a))
            if long_pas <= 0.0:
                continue
            milieu = v0 + pente * (0.5 * (max(u0, float(a)) + min(u1, a + 1.0)) - u0)
            debut, fin = milieu - demi, milieu + demi
            for b in range(max(0, int(np.floor(debut))), min(cote, int(np.ceil(fin)))):
                cache = (min(fin, b + 1.0) - max(debut, float(b))) * long_pas
                visible = np.uint8(max(0.0, 255.0 * (1.0 - cache)) + 0.5)
                if horizontal:
                    if visible < dessin[b, a]:
                        dessin[b, a] = visible
                else:
                    if visible < dessin[a, b]:
                        dessin[a, b] = visible


@njit(cache=True)
def _reduire(dessin, facteur):
    """Réduit par moyenne de blocs `facteur × facteur` ; rend la part visible dans [0, 1]."""
    cote = dessin.shape[0] // facteur
    sortie = np.empty((cote, cote), np.float32)
    for i in range(cote):
        for j in range(cote):
            somme = 0
            for a in range(i * facteur, (i + 1) * facteur):
                for b in range(j * facteur, (j + 1) * facteur):
                    somme += dessin[a, b]
            sortie[i, j] = somme / (255.0 * facteur * facteur)
    return sortie


def rendu_physique(
    toile: Toile, sequence, taille: int | None = None, gamma: float = GAMMA
) -> np.ndarray:
    """Simule la toile montée : image perçue `taille × taille` (par défaut `toile.cote`).

    `sequence` est la suite de clous ; tous les pas sont dessinés, y compris ceux du bord.
    Chaque fil est un trait opaque de la largeur réelle, dessiné sur une image assez grande
    pour qu'il y fasse environ trois pixels. La réduction moyenne la *lumière* (c'est ce que
    fait l'œil ou un appareil photo), puis le résultat est codé en valeur perçue : l'image
    rendue se regarde et se compare comme une photo de la pièce. `gamma=1` rend la lumière brute.
    """
    taille = toile.cote if taille is None else int(taille)
    seq = np.asarray(sequence, dtype=np.int64)
    fils_par_cote = toile.diametre_mm / toile.epaisseur_fil_mm
    facteur = int(np.ceil(_PIXELS_PAR_FIL * fils_par_cote / taille))
    facteur = max(1, min(facteur, _COTE_DESSIN_MAX // taille))
    cote_dessin = taille * facteur

    dessin = np.full((cote_dessin, cote_dessin), 255, np.uint8)
    if seq.shape[0] >= 2:
        points = positions_clous(toile.n_clous, cote_dessin)[seq]
        _tracer_fils(dessin, points, 0.5 * cote_dessin / fils_par_cote)
    lumiere = _reduire(dessin, facteur)
    return lumiere if gamma == 1.0 else lumiere ** np.float32(1.0 / gamma)


def en_image(luminance: np.ndarray, taille: int | None = None) -> Image.Image:
    """Convertit une luminance [0, 1] en image Pillow en niveaux de gris."""
    image = Image.fromarray((np.clip(luminance, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "L")
    if taille is not None and image.size != (taille, taille):
        image = image.resize((taille, taille), Image.Resampling.LANCZOS)
    return image


def planche(images: list[np.ndarray | Image.Image], taille: int = 400, marge: int = 8):
    """Juxtapose des images (luminances ou images Pillow) sur une ligne, sur fond blanc."""
    vignettes = [
        (im if isinstance(im, Image.Image) else en_image(im))
        .convert("L")
        .resize((taille, taille), Image.Resampling.LANCZOS)
        for im in images
    ]
    largeur = len(vignettes) * taille + (len(vignettes) + 1) * marge
    feuille = Image.new("L", (largeur, taille + 2 * marge), 255)
    for i, vignette in enumerate(vignettes):
        feuille.paste(vignette, (marge + i * (taille + marge), marge))
    return feuille


def svg(toile: Toile, sequence, numeros_tous_les: int = 10) -> str:
    """La pièce en SVG, à l'échelle 1 (unités en millimètres) : fil, clous et repères numérotés.

    Imprimé sans mise à l'échelle, le dessin sert de gabarit pour planter les clous.
    """
    d = toile.diametre_mm
    marge = max(12.0, d * 0.04)
    cote = d + 2 * marge
    # Mêmes conventions que positions_clous, en millimètres et décalées de la marge.
    points = positions_clous(toile.n_clous, 1) * d + marge
    seq = [int(c) for c in sequence]
    trace = " ".join(f"{points[c, 1]:.2f},{points[c, 0]:.2f}" for c in seq)
    lignes = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{cote:.1f}mm" height="{cote:.1f}mm" '
        f'viewBox="0 0 {cote:.1f} {cote:.1f}">',
        f'<rect width="{cote:.1f}" height="{cote:.1f}" fill="white"/>',
        f'<circle cx="{cote / 2:.2f}" cy="{cote / 2:.2f}" r="{d / 2:.2f}" fill="none" '
        'stroke="#bbb" stroke-width="0.2"/>',
    ]
    if len(seq) >= 2:
        lignes.append(
            f'<polyline points="{trace}" fill="none" stroke="black" '
            f'stroke-width="{toile.epaisseur_fil_mm:g}" stroke-linejoin="round"/>'
        )
    rayon_clou = max(0.5, min(1.0, np.pi * d / toile.n_clous / 5))
    centre = cote / 2
    for i, (r, c) in enumerate(points):
        lignes.append(f'<circle cx="{c:.2f}" cy="{r:.2f}" r="{rayon_clou:.2f}" fill="#c00"/>')
        if i % numeros_tous_les == 0:
            # Le numéro est poussé vers l'extérieur, dans le prolongement du rayon.
            dx, dy = c - centre, r - centre
            norme = float(np.hypot(dx, dy))
            x = c + dx / norme * marge * 0.5
            y = r + dy / norme * marge * 0.5
            lignes.append(
                f'<text x="{x:.2f}" y="{y:.2f}" font-size="{marge * 0.3:.1f}" '
                'font-family="sans-serif" text-anchor="middle" dominant-baseline="middle">'
                f"{i}</text>"
            )
    lignes.append("</svg>")
    return "\n".join(lignes) + "\n"
