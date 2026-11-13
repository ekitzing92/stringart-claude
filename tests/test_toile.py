import numpy as np
import pytest

from stringart_claude.toile import (
    GAMMA,
    Geometrie,
    Toile,
    geometrie,
    masque_disque,
    positions_clous,
)


def test_toile():
    # Les grandeurs dérivées relient les millimètres aux pixels
    t = Toile(diametre_mm=500.0, n_clous=240, epaisseur_fil_mm=0.2, cote=250)
    assert t.pas_mm == pytest.approx(2.0)
    assert t.alpha == pytest.approx(0.1)

    # L'écart minimal par défaut suit le nombre de clous, sans descendre sous 2
    assert Toile(n_clous=240).ecart == 10
    assert Toile(n_clous=24).ecart == 2
    assert Toile(n_clous=240, ecart_min=1).ecart == 2

    # Les valeurs absurdes sont refusées
    for arguments in ({"n_clous": 4}, {"cote": 8}, {"diametre_mm": 0}, {"epaisseur_fil_mm": -1}):
        with pytest.raises(ValueError):
            Toile(**arguments)
    with pytest.raises(ValueError):
        Toile(n_clous=20, ecart_min=10)


def test_positions_clous():
    pos = positions_clous(8, 100)

    # Le clou 0 est en haut, au milieu
    assert pos[0] == pytest.approx([0.0, 50.0])

    # Les numéros croissent dans le sens horaire : le clou n/4 est à droite, n/2 en bas
    assert pos[2] == pytest.approx([50.0, 100.0])
    assert pos[4] == pytest.approx([100.0, 50.0])
    assert pos[6] == pytest.approx([50.0, 0.0])

    # Tous les clous sont sur le cercle inscrit
    assert np.hypot(pos[:, 0] - 50, pos[:, 1] - 50) == pytest.approx(np.full(8, 50.0))


def test_masque_disque():
    masque = masque_disque(64)

    # Le centre est dedans, les coins dehors
    assert masque[32, 32] and not masque[0, 0] and not masque[63, 63]

    # Le disque touche les quatre bords et couvre π/4 du carré
    assert masque[0].any() and masque[-1].any() and masque[:, 0].any() and masque[:, -1].any()
    assert masque.mean() == pytest.approx(np.pi / 4, abs=0.01)


def test_geometrie(toile, geo):
    n = toile.n_clous

    # Une corde par paire de clous assez éloignés, numérotée dans les deux sens
    assert geo.n_cordes == n * (n - 2 * toile.ecart + 1) // 2
    assert (geo.id_corde == geo.id_corde.T).all()
    assert geo.id_corde[0, 1] == -1 and geo.id_corde[0, n // 2] >= 0
    k = geo.id_corde[3, 20]
    assert (geo.clou_a[k], geo.clou_b[k]) == (3, 20)

    # La couverture totale d'une corde vaut sa longueur (en pixels) fois l'épaisseur du fil.
    # (Vérifié sur les cordes longues : les courtes perdent une part notable de leur longueur
    # dans les pixels du pourtour, dont le centre est hors du disque.)
    couverture = np.asarray(geo.matrice_couverture().sum(axis=0)).ravel()
    attendue = toile.alpha * geo.longueur_mm / toile.pas_mm
    longues = geo.clou_b - geo.clou_a == n // 2
    assert couverture[longues] == pytest.approx(attendue[longues], rel=0.03)
    assert (couverture <= attendue * 1.001).all()

    # Aucune corde ne déborde du disque
    assert geo.masque.ravel()[geo.base_pix].all()

    # Le rangement par pixel contient exactement les mêmes coefficients que celui par corde
    assert geo.p_corde.shape == geo.c_pix.shape
    assert np.sort(geo.p_q) == pytest.approx(np.sort(geo.c_q))
    p = int(geo.c_pix[geo.c_ptr[k]])
    assert k in geo.p_corde[geo.p_ptr[p] : geo.p_ptr[p + 1]]

    # Le moteur reçoit q à la puissance 1/gamma ; la matrice physique reste intacte
    assert geo.gamma == GAMMA
    assert geo.c_q == pytest.approx(geo.base_q.astype(np.float64) ** (1 / GAMMA), rel=1e-5)
    sans_gamma = Geometrie(toile, gamma=1.0)
    assert sans_gamma.c_q == pytest.approx(sans_gamma.base_q)

    # Avec plusieurs échelles, les cellules grossières s'ajoutent à la suite des pixels
    multi = Geometrie(toile, echelles=(1, 4), gamma=1.0)
    assert multi.n_cellules == 64 * 64 + 16 * 16
    assert [niveau[:2] for niveau in multi.niveaux] == [(1, 64), (4, 16)]
    debut, fin = multi.c_ptr[k], multi.c_ptr[k + 1]
    grossier = multi.c_pix[debut:fin] >= 64 * 64
    # ... et une cellule entièrement dans le disque reçoit la moyenne de ses 16 pixels
    assert (1 - multi.c_q[debut:fin][grossier]).sum() * 16 == pytest.approx(
        (1 - multi.c_q[debut:fin][~grossier]).sum(), rel=0.1
    )

    # Les paramètres invalides sont refusés
    with pytest.raises(ValueError):
        Geometrie(toile, echelles=(2, 4))
    with pytest.raises(ValueError):
        Geometrie(toile, gamma=0.0)


def test_regrouper(geo):
    champ = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)

    # La somme totale est conservée, et chaque bloc contient bien ses pixels
    blocs = geo.regrouper(champ, 4)
    assert blocs.shape == (16, 16)
    assert blocs.sum() == pytest.approx(champ.sum())
    assert blocs[1, 2] == pytest.approx(champ[4:8, 8:12].sum())

    # Un bloc qui ne tombe pas juste est complété par des zéros
    assert geo.regrouper(champ, 5).shape == (13, 13)
    assert geo.regrouper(champ, 5).sum() == pytest.approx(champ.sum())


def test_cordes_de_sequence(geo):
    # Une séquence rend ses cordes dans l'ordre ; les pas de bord n'en sont pas
    sequence = [0, 20, 21, 40, 5]
    attendu = [geo.id_corde[0, 20], geo.id_corde[21, 40], geo.id_corde[40, 5]]
    assert geo.cordes_de_sequence(sequence).tolist() == attendu

    # Une séquence trop courte ne contient aucune corde
    assert geo.cordes_de_sequence([3]).shape == (0,)


def test_geometrie_cache(toile, geo):
    # Deux toiles égales partagent la même géométrie
    assert geometrie(Toile(diametre_mm=200.0, n_clous=48, epaisseur_fil_mm=0.4, cote=64)) is geo
