import numpy as np
import pytest

from stringart_claude.evaluation import comparer, evaluer
from stringart_claude.rendu import rendu_physique


def test_comparer(cible):
    # Une image comparée à elle-même est parfaite
    notes = comparer(cible, cible)
    assert notes.eqm == pytest.approx(0.0) and notes.eqm_floue == pytest.approx(0.0)
    assert notes.ssim == pytest.approx(1.0)

    # Un décalage uniforme se retrouve tel quel dans les deux écarts
    plus_clair = comparer(cible, cible + 0.1)
    assert plus_clair.eqm == pytest.approx(0.1, abs=1e-3)
    assert plus_clair.eqm_floue == pytest.approx(0.1, abs=5e-3)

    # Un grain fin pèse sur l'écart brut bien plus que sur l'écart flou
    # (sur une image plus grande : le flou est proportionnel à la taille)
    hasard = np.random.default_rng(5)
    grande = np.full((200, 200), 0.5)
    bruitee = comparer(grande, grande + 0.1 * hasard.standard_normal(grande.shape))
    assert bruitee.eqm > 3 * bruitee.eqm_floue and bruitee.ssim < 0.9

    # Ce qui se passe hors du disque ne compte pas
    dehors = cible.copy()
    dehors[0, 0] = 0.0
    assert comparer(cible, dehors).eqm == pytest.approx(0.0)

    # Les notes se rendent en dictionnaire ; les formes différentes sont refusées
    assert set(notes.dict()) == {"eqm", "eqm_floue", "ssim"}
    with pytest.raises(ValueError):
        comparer(cible, cible[:32, :32])


def test_evaluer(toile, cible):
    # Évaluer une séquence, c'est comparer sa simulation à la cible
    sequence = [0, 24, 5, 30, 11, 40]
    assert evaluer(toile, sequence, cible) == comparer(cible, rendu_physique(toile, sequence))
