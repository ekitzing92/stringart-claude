"""Petites toiles partagées par les tests : assez grandes pour être réalistes, assez petites
pour que chaque solveur tourne en une fraction de seconde."""

import numpy as np
import pytest

from stringart_claude.moteur import Probleme
from stringart_claude.toile import Toile, geometrie


@pytest.fixture(scope="session")
def toile() -> Toile:
    return Toile(diametre_mm=200.0, n_clous=48, epaisseur_fil_mm=0.4, cote=64)


@pytest.fixture(scope="session")
def geo(toile):
    return geometrie(toile)


@pytest.fixture(scope="session")
def cible(toile) -> np.ndarray:
    """Un disque sombre décentré sur fond clair, à bords doux."""
    axe = np.arange(toile.cote) + 0.5
    distance = np.hypot(axe[:, None] - 26.0, axe[None, :] - 36.0)
    return (0.35 + 0.6 / (1.0 + np.exp(-(distance - 12.0) / 2.0))).astype(np.float32)


@pytest.fixture(scope="session")
def probleme(geo, cible) -> Probleme:
    return Probleme(geo, cible)
