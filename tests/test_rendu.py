import numpy as np
import pytest
from PIL import Image

from stringart_claude.moteur import image_modele
from stringart_claude.rendu import en_image, planche, rendu_physique, svg
from stringart_claude.toile import GAMMA


def test_rendu_physique(toile, geo):
    # Sans fil, la toile est blanche
    assert rendu_physique(toile, []) == pytest.approx(1.0)
    assert rendu_physique(toile, [3]).shape == (64, 64)

    # Un fil cache une surface égale à sa longueur fois son épaisseur, quelle que soit sa pente
    for a, b in [(0, 24), (3, 20), (10, 31), (40, 17)]:
        lumiere = rendu_physique(toile, [a, b], gamma=1.0)
        k = geo.id_corde[a, b]
        attendue = toile.alpha * geo.longueur_mm[k] / toile.pas_mm
        assert (1.0 - lumiere).sum() == pytest.approx(attendue, rel=0.02)

    # L'image rendue est la lumière codée en valeur perçue
    sequence = [0, 24, 5, 30, 11, 40]
    lumiere = rendu_physique(toile, sequence, gamma=1.0)
    assert rendu_physique(toile, sequence) == pytest.approx(lumiere ** (1 / GAMMA), abs=1e-6)

    # Simulation et modèle des solveurs s'accordent en moyenne sur une toile bien remplie
    hasard = np.random.default_rng(4)
    cordes = hasard.choice(geo.n_cordes, 120, replace=False)
    sequence = np.stack([geo.clou_a[cordes], geo.clou_b[cordes]], axis=1)
    simulation = np.ones((64, 64))
    for paire in sequence:
        simulation *= rendu_physique(toile, paire, gamma=1.0)
    modele = image_modele(geo, cordes).astype(np.float64) ** GAMMA
    assert simulation[geo.masque].mean() == pytest.approx(modele[geo.masque].mean(), abs=0.01)

    # La taille de sortie est libre, et l'encre totale n'en dépend pas
    grand = rendu_physique(toile, [0, 24, 5, 30], taille=160, gamma=1.0)
    petit = rendu_physique(toile, [0, 24, 5, 30], taille=64, gamma=1.0)
    assert grand.shape == (160, 160)
    assert (1 - grand).mean() == pytest.approx((1 - petit).mean(), rel=0.03)


def test_en_image():
    luminance = np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float32)

    # Les valeurs sont bornées puis ramenées sur 8 bits
    image = en_image(luminance)
    assert image.mode == "L" and image.size == (2, 2)
    assert np.asarray(image).ravel().tolist() == [0, 128, 255, 255]

    # La taille demandée est respectée
    assert en_image(luminance, taille=8).size == (8, 8)


def test_planche():
    blanc = np.ones((20, 20), dtype=np.float32)
    noir = Image.new("L", (30, 30), 0)

    # Tableaux et images Pillow se mélangent, chacun dans sa vignette
    feuille = planche([blanc, noir], taille=40, marge=4)
    assert feuille.size == (2 * 40 + 3 * 4, 40 + 2 * 4)
    assert feuille.getpixel((10, 10)) == 255 and feuille.getpixel((70, 20)) == 0


def test_svg(toile):
    texte = svg(toile, [0, 24, 5, 30])

    # Le dessin est à l'échelle 1, en millimètres
    assert 'width="224.0mm"' in texte and texte.startswith("<svg")

    # Le fil est une seule ligne brisée de la bonne épaisseur, les clous sont tous là
    assert texte.count("<polyline") == 1 and 'stroke-width="0.4"' in texte
    assert texte.count('fill="#c00"') == toile.n_clous

    # Un clou sur dix est numéroté
    assert ">0</text>" in texte and ">40</text>" in texte and ">5</text>" not in texte

    # Sans séquence, il reste le gabarit de perçage
    assert "<polyline" not in svg(toile, [])
