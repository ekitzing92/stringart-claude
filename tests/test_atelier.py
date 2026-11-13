import numpy as np
import pytest

from stringart_claude.atelier import Oeuvre, creer
from stringart_claude.evaluation import comparer
from stringart_claude.solveurs import RECOMMANDE


@pytest.fixture(scope="module")
def oeuvre(toile, cible) -> Oeuvre:
    return creer(cible, toile, contraste="aucun", noir=0.0)


def test_creer(toile, cible, oeuvre, tmp_path):
    # La chaîne rend la cible, la solution de la méthode recommandée et la simulation notée
    assert oeuvre.toile is toile and oeuvre.resultat.nom == RECOMMANDE
    assert oeuvre.cible.shape == (64, 64) and oeuvre.rendu.shape == (64, 64)
    assert oeuvre.notes == comparer(oeuvre.cible, oeuvre.rendu)
    assert oeuvre.resultat.n_cordes > 0 and oeuvre.fil_m > 0

    # Un fichier image est accepté comme un tableau, et les réglages d'image sont appliqués
    from PIL import Image

    Image.fromarray((cible * 255).astype(np.uint8)).save(tmp_path / "cible.png")
    depuis_fichier = creer(tmp_path / "cible.png", toile, contraste="aucun", noir=0.0)
    assert depuis_fichier.cible == pytest.approx(oeuvre.cible, abs=0.01)
    assert creer(cible, toile, contraste="aucun", noir=0.5).cible.min() >= 0.5

    # Le solveur et ses options sont transmis
    courte = creer(cible, toile, solveur="glouton_global", cordes_max=5)
    assert courte.resultat.nom == "glouton_global" and courte.resultat.n_cordes == 5


def test_oeuvre(oeuvre, tmp_path):
    # L'aperçu se rend à la taille voulue, le résumé dit l'essentiel
    assert oeuvre.apercu(120).shape == (120, 120)
    resume = oeuvre.resume()
    assert f"{oeuvre.resultat.n_cordes} cordes" in resume and "m de fil" in resume
    assert oeuvre.sequence is oeuvre.resultat.sequence

    # L'enregistrement crée le dossier et écrit les cinq livrables
    chemins = oeuvre.enregistrer(tmp_path / "piece", taille=200)
    assert set(chemins) == {"rendu", "comparaison", "sequence", "guide", "svg"}
    assert all(chemin.stat().st_size > 0 for chemin in chemins.values())
    assert "Départ : nouer le fil" in chemins["sequence"].read_text("utf-8")
