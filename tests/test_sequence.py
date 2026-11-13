import numpy as np
import pytest

from stringart_claude.sequence import (
    est_pas_de_bord,
    longueur_fil_m,
    sequencer,
    texte_sequence,
)
from stringart_claude.toile import Toile


def _verifier(geo, cordes, sequence):
    """Une séquence valide passe une fois par chaque corde ; le reste est du bord."""
    n = geo.toile.n_clous
    parcourues = geo.cordes_de_sequence(sequence)
    assert sorted(parcourues.tolist()) == sorted(int(k) for k in cordes)
    bord = est_pas_de_bord(sequence, n)
    assert bord.sum() + len(parcourues) == len(sequence) - 1


def test_sequencer(geo):
    hasard = np.random.default_rng(1)

    # Un ensemble quelconque de cordes devient un fil continu
    cordes = hasard.choice(geo.n_cordes, 150, replace=False)
    _verifier(geo, cordes, sequencer(geo, cordes))

    # Un masque booléen est accepté comme une liste
    masque = np.zeros(geo.n_cordes, bool)
    masque[cordes] = True
    _verifier(geo, cordes, sequencer(geo, masque))

    # Un parcours déjà eulérien n'a besoin d'aucun pas de bord
    triangle = [geo.id_corde[0, 16], geo.id_corde[16, 32], geo.id_corde[32, 0]]
    sequence = sequencer(geo, triangle)
    assert len(sequence) == 4 and not est_pas_de_bord(sequence, 48).any()

    # Deux cordes sans clou commun sont reliées par le bord, au plus court
    isolees = [geo.id_corde[0, 20], geo.id_corde[22, 40]]
    sequence = sequencer(geo, isolees)
    _verifier(geo, isolees, sequence)
    assert len(sequence) - 1 == 2 + 2  # deux cordes, deux pas de bord entre 20 et 22

    # Deux îlots dont tous les clous sont pairs : un aller-retour le long du bord les relie
    triangles = triangle + [geo.id_corde[2, 18], geo.id_corde[18, 34], geo.id_corde[34, 2]]
    sequence = sequencer(geo, triangles)
    _verifier(geo, triangles, sequence)
    assert est_pas_de_bord(sequence, 48).sum() == 2 * 2  # de 0 à 2 et retour

    # Une seule corde, et aucune corde
    assert sorted(sequencer(geo, [geo.id_corde[5, 30]]).tolist()) == [5, 30]
    assert sequencer(geo, []).shape == (0,)


def test_est_pas_de_bord():
    # Clous voisins, y compris à travers le clou 0
    assert est_pas_de_bord([3, 4, 3, 10, 47, 0], 48).tolist() == [True, True, False, False, True]


def test_longueur_fil_m():
    toile = Toile(diametre_mm=1000.0, n_clous=8, cote=100, ecart_min=2)

    # Un diamètre mesure un mètre ; l'aller-retour, deux
    assert longueur_fil_m(toile, [0, 4]) == pytest.approx(1.0)
    assert longueur_fil_m(toile, [0, 4, 0]) == pytest.approx(2.0)

    # Une séquence d'un seul clou ne consomme pas de fil
    assert longueur_fil_m(toile, [3]) == 0.0


def test_texte_sequence(toile):
    sequence = [0, 20, 21, 40, 5, 30, 10, 35, 15, 44, 2, 25, 1]
    texte = texte_sequence(toile, sequence)

    # L'en-tête rappelle la convention de numérotation et le point de départ
    assert "Clou 0 en haut" in texte
    assert "Départ : nouer le fil au clou 0." in texte
    assert "12 passages (1 le long du bord)" in texte

    # Dix clous par ligne, précédés du rang des passages
    lignes = [ligne for ligne in texte.splitlines() if "|" in ligne]
    assert len(lignes) == 2
    assert lignes[0].split("|")[0].strip() == "1-10"
    assert lignes[0].split("|")[1].split() == [str(c) for c in sequence[1:11]]
    assert lignes[1].split("|")[0].strip() == "11-12"

    # Une séquence vide se dit telle quelle
    assert "(séquence vide)" in texte_sequence(toile, [])
