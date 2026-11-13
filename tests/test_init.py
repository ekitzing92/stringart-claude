import stringart_claude


def test_init():
    # Tout ce que le paquet annonce existe bel et bien
    for nom in stringart_claude.__all__:
        assert hasattr(stringart_claude, nom), nom

    # Le chemin court de la documentation est exposé à la racine
    assert callable(stringart_claude.creer) and stringart_claude.Toile().n_clous == 240
