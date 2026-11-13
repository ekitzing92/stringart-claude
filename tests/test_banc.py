import pytest

from stringart_claude import banc
from stringart_claude.solveurs import glouton_global


@pytest.fixture(scope="module")
def mesures(probleme):
    resultats = banc.lancer(
        {"disque": probleme}, ["glouton_global", "tri_naif"], {"tri_naif": {"cordes_max": 20}},
        garder_resultats=True,
    )  # fmt: skip
    return resultats


def test_mesurer(probleme):
    resultat = glouton_global(probleme)
    mesure = banc.mesurer("disque", probleme, resultat)

    # La mesure reprend le résultat et y ajoute le juge et le coût de montage
    assert (mesure.probleme, mesure.solveur) == ("disque", "glouton_global")
    assert mesure.ecart_modele == pytest.approx(resultat.cout**0.5)
    assert mesure.cordes == resultat.n_cordes
    assert mesure.cordes + mesure.pas_de_bord == len(resultat.sequence) - 1
    assert mesure.fil_m > 0 and 0 < mesure.eqm_floue < mesure.eqm and mesure.ssim <= 1


def test_lancer(mesures, probleme):
    liste, resultats = mesures

    # Une mesure par couple (problème, solveur), dans l'ordre demandé
    assert [m.solveur for m in liste] == ["glouton_global", "tri_naif"]
    assert set(resultats) == {("disque", "glouton_global"), ("disque", "tri_naif")}

    # Les options propres à un solveur lui sont transmises
    assert liste[1].cordes == 20

    # Par défaut, les résultats complets ne sont pas conservés
    _, vide = banc.lancer({"disque": probleme}, ["tri_naif"])
    assert vide == {}


def test_classement(mesures):
    lignes = banc.classement(mesures[0])

    # Le meilleur solveur est en tête, avec un retard nul sur lui-même
    assert [ligne["solveur"] for ligne in lignes] == ["glouton_global", "tri_naif"]
    assert lignes[0]["retard_pct"] == pytest.approx(0.0)
    assert lignes[1]["retard_pct"] > 0


def test_tableau_mesures(mesures):
    # Un en-tête, un séparateur, une ligne par mesure
    lignes = banc.tableau_mesures(mesures[0]).splitlines()
    assert len(lignes) == 4 and lignes[2].startswith("| disque | glouton_global |")


def test_tableau_classement(mesures):
    # Un en-tête, un séparateur, une ligne par solveur ; le premier a un retard nul
    lignes = banc.tableau_classement(mesures[0]).splitlines()
    assert len(lignes) == 4 and "+0.00 %" in lignes[2]
