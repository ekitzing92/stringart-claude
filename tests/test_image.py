import numpy as np
import pytest
from PIL import Image

from stringart_claude.image import charger_image, preparer_cible
from stringart_claude.toile import masque_disque


def test_charger_image(tmp_path):
    # Une image en couleurs devient une luminance entre 0 et 1
    Image.new("RGB", (6, 4), (255, 255, 255)).save(tmp_path / "blanc.png")
    blanc = charger_image(tmp_path / "blanc.png")
    assert blanc.shape == (4, 6) and blanc.dtype == np.float32
    assert blanc == pytest.approx(1.0)

    # La transparence est composée sur du blanc, pas sur du noir
    Image.new("RGBA", (5, 5), (0, 0, 0, 0)).save(tmp_path / "vide.png")
    assert charger_image(tmp_path / "vide.png") == pytest.approx(1.0)

    # Un fichier absent lève l'erreur habituelle
    with pytest.raises(OSError):
        charger_image(tmp_path / "absent.png")


def test_preparer_cible():
    disque = masque_disque(40)
    degrade = np.tile(np.linspace(0.2, 0.8, 120, dtype=np.float32), (80, 1))

    # La cible est carrée, à la résolution demandée, blanche hors du disque
    cible = preparer_cible(degrade, 40, contraste="aucun", noir=0.0)
    assert cible.shape == (40, 40) and cible.dtype == np.float32
    assert cible[~disque] == pytest.approx(1.0)

    # Une image en paysage est recadrée au carré central : on ne voit que le milieu du dégradé
    assert cible[20, 2] == pytest.approx(0.2 + 0.6 * 22 / 120, abs=0.02)
    assert cible[20, 37] == pytest.approx(0.2 + 0.6 * 98 / 120, abs=0.02)

    # Le zoom resserre le cadrage, le décalage le déplace
    serre = preparer_cible(degrade, 40, contraste="aucun", noir=0.0, zoom=2.0)
    assert serre[20, 2] > cible[20, 2] and serre[20, 37] < cible[20, 37]
    droite = preparer_cible(degrade, 40, contraste="aucun", noir=0.0, decalage=(0.2, 0.0))
    assert droite[20, 20] > cible[20, 20]

    # Ce qui sort de l'image d'origine est complété en blanc
    large = preparer_cible(
        np.zeros((50, 50), np.float32), 40, contraste="aucun", noir=0.0, zoom=0.5
    )
    assert large[20, 20] == pytest.approx(0.0) and large[20, 3] == pytest.approx(1.0)

    # Le contraste automatique étire les tons sur toute la plage
    etire = preparer_cible(degrade, 40, contraste="auto", noir=0.0)
    assert etire[disque].min() < 0.05 and etire[disque].max() > 0.95

    # Le contraste local garde des valeurs valides ; sans relief, c'est le contraste automatique
    local = preparer_cible(degrade, 40, contraste="local", noir=0.0)
    assert 0.0 <= local.min() and local.max() <= 1.0
    plat = preparer_cible(degrade, 40, contraste="local", relief=0.0, noir=0.0)
    assert plat == pytest.approx(etire)

    # `noir` relève le plancher des tons, `gamma` assombrit les tons moyens
    releve = preparer_cible(degrade, 40, contraste="auto", noir=0.3)
    assert releve[disque].min() == pytest.approx(0.3, abs=0.03)
    sombre = preparer_cible(degrade, 40, contraste="aucun", noir=0.0, gamma=2.0)
    assert sombre[20, 20] == pytest.approx(cible[20, 20] ** 2, abs=0.01)

    # Les réglages invalides sont refusés
    with pytest.raises(ValueError):
        preparer_cible(degrade, 40, contraste="fort")
    with pytest.raises(ValueError):
        preparer_cible(degrade, 40, zoom=0)
    with pytest.raises(ValueError):
        preparer_cible(degrade, 40, gamma=-1)
    with pytest.raises(ValueError):
        preparer_cible(degrade, 40, noir=1.0)
    with pytest.raises(ValueError):
        preparer_cible(degrade, 40, relief=2.0)
    with pytest.raises(ValueError):
        preparer_cible(np.zeros((4, 4, 3)), 40)
