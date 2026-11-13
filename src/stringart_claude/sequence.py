"""D'un ensemble de cordes à un fil continu, et sa mise en forme pour le montage.

L'image finale ne dépend pas de l'ordre de pose : seul compte *l'ensemble* des cordes. On peut
donc laisser les solveurs choisir librement leurs cordes, puis chercher après coup un parcours
d'un seul tenant. C'est un problème de **chemin eulérien** sur le graphe dont les sommets sont
les clous et les arêtes les cordes.

Un tel chemin n'existe que si (presque) tous les clous ont un nombre pair de cordes. Pour
rétablir la parité on s'autorise un geste gratuit pour l'image : **longer le bord**, c'est-à-dire
passer d'un clou à son voisin par l'extérieur. Ces pas ne traversent pas la toile. Les clous de
degré impair sont appariés de proche en proche autour du cercle — sur un cercle, l'appariement
optimal est l'un des deux appariements de voisins consécutifs — et reliés par ces pas de bord.

Dans une séquence, un pas de bord se reconnaît à deux numéros consécutifs.
"""

from __future__ import annotations

import numpy as np

from .toile import Geometrie, Toile


def sequencer(geo: Geometrie, cordes) -> np.ndarray:
    """Rend une suite de clous parcourant chaque corde exactement une fois, d'un seul fil.

    `cordes` : numéros de cordes (sans doublon) ou masque booléen. La suite peut contenir des
    pas de bord (clous voisins) ajoutés pour rendre le parcours possible.
    """
    cordes = np.asarray(cordes)
    if cordes.dtype == np.bool_:
        cordes = np.flatnonzero(cordes)
    if cordes.shape[0] == 0:
        return np.empty(0, np.int32)
    n = geo.toile.n_clous

    aretes_u = list(geo.clou_a[cordes].tolist())
    aretes_v = list(geo.clou_b[cordes].tolist())

    def ajouter_bord(u: int, v: int) -> None:
        """Ajoute les pas de bord de u jusqu'à v en tournant dans le sens horaire."""
        while u != v:
            suivant = (u + 1) % n
            aretes_u.append(u)
            aretes_v.append(suivant)
            u = suivant

    # 1. Connexité : si les cordes forment plusieurs îlots, un tour complet du bord les relie
    #    (et ajoute 2 au degré de chaque clou, donc ne change aucune parité).
    if _n_composantes(n, aretes_u, aretes_v) > 1:
        ajouter_bord(0, n - 1)
        aretes_u.append(n - 1)
        aretes_v.append(0)

    # 2. Parité : apparier les clous impairs voisins. Deux appariements possibles sur un cercle,
    #    décalés d'un cran ; on garde le plus court, et on laisse ouverte sa plus longue paire :
    #    ses deux clous seront le départ et l'arrivée du fil.
    degre = np.bincount(aretes_u, minlength=n) + np.bincount(aretes_v, minlength=n)
    impairs = np.flatnonzero(degre % 2 == 1)
    depart = int(aretes_u[0])
    if impairs.shape[0] > 0:
        arcs = np.diff(np.append(impairs, impairs[0] + n))  # arc de chaque impair au suivant
        decalage = 0 if arcs[0::2].sum() <= arcs[1::2].sum() else 1
        paires = [
            (int(impairs[i]), int(impairs[(i + 1) % impairs.shape[0]]), int(arcs[i]))
            for i in range(decalage, impairs.shape[0], 2)
        ]
        ouverte = max(range(len(paires)), key=lambda i: paires[i][2])
        for i, (u, v, _) in enumerate(paires):
            if i != ouverte:
                ajouter_bord(u, v)
        depart = paires[ouverte][0]

    return _hierholzer(n, aretes_u, aretes_v, depart)


def _n_composantes(n: int, aretes_u: list[int], aretes_v: list[int]) -> int:
    """Nombre de composantes connexes parmi les clous qui portent au moins une arête."""
    parent = list(range(n))

    def racine(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in zip(aretes_u, aretes_v, strict=True):
        parent[racine(u)] = racine(v)
    return len({racine(x) for x in set(aretes_u) | set(aretes_v)})


def _hierholzer(n: int, aretes_u: list[int], aretes_v: list[int], depart: int) -> np.ndarray:
    """Chemin eulérien par l'algorithme de Hierholzer (version itérative)."""
    voisins: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for e, (u, v) in enumerate(zip(aretes_u, aretes_v, strict=True)):
        voisins[u].append((v, e))
        voisins[v].append((u, e))
    utilisee = [False] * len(aretes_u)
    pile = [depart]
    chemin: list[int] = []
    while pile:
        u = pile[-1]
        while voisins[u] and utilisee[voisins[u][-1][1]]:
            voisins[u].pop()
        if voisins[u]:
            v, e = voisins[u].pop()
            utilisee[e] = True
            pile.append(v)
        else:
            chemin.append(pile.pop())
    if len(chemin) != len(aretes_u) + 1:
        raise RuntimeError("parcours eulérien incomplet : graphe non connexe ou parité fausse")
    return np.asarray(chemin[::-1], dtype=np.int32)


def est_pas_de_bord(sequence, n_clous: int) -> np.ndarray:
    """Pour chaque pas de la séquence, vrai s'il longe le bord (clous voisins)."""
    seq = np.asarray(sequence, dtype=np.int64)
    ecart = np.abs(np.diff(seq)) % n_clous
    return np.minimum(ecart, n_clous - ecart) == 1


def longueur_fil_m(toile: Toile, sequence) -> float:
    """Longueur de fil consommée par la séquence, en mètres (hors enroulement sur les clous)."""
    from .toile import positions_clous

    seq = np.asarray(sequence, dtype=np.int64)
    if seq.shape[0] < 2:
        return 0.0
    points = positions_clous(toile.n_clous, toile.cote)[seq]
    pas = np.hypot(*(points[1:] - points[:-1]).T)
    return float(pas.sum() * toile.pas_mm / 1000.0)


def texte_sequence(toile: Toile, sequence, par_ligne: int = 10) -> str:
    """La séquence sous une forme qui se suit à la main, des heures durant.

    Dix clous par ligne, précédés du rang des passages couverts : on retrouve sa place d'un coup
    d'œil après une pause. L'en-tête rappelle la convention de numérotation, sans laquelle les
    numéros ne désignent rien de physique.
    """
    seq = [int(c) for c in sequence]
    n_pas = max(len(seq) - 1, 0)
    n_bord = int(est_pas_de_bord(seq, toile.n_clous).sum()) if n_pas else 0
    largeur = len(str(toile.n_clous - 1))
    largeur_rang = len(str(max(n_pas, 1)))
    lignes = [
        "STRING ART — séquence de montage",
        "",
        f"Toile ronde de {toile.diametre_mm:g} mm, {toile.n_clous} clous régulièrement espacés,"
        f" fil de {toile.epaisseur_fil_mm:g} mm.",
        "Clou 0 en haut ; les numéros croissent dans le sens des aiguilles d'une montre.",
        f"{n_pas} passages ({n_bord} le long du bord), "
        f"environ {longueur_fil_m(toile, seq):.0f} m de fil (prévoir 10 % de marge).",
        "Deux numéros consécutifs (ex. 41 puis 42) : longer le bord jusqu'au clou voisin.",
        "",
    ]
    if not seq:
        return "\n".join(lignes + ["(séquence vide)"]) + "\n"
    lignes += [f"Départ : nouer le fil au clou {seq[0]}.", ""]
    suite = seq[1:]
    for debut in range(0, len(suite), par_ligne):
        bloc = suite[debut : debut + par_ligne]
        rang = f"{debut + 1:>{largeur_rang}}-{debut + len(bloc):<{largeur_rang}}"
        lignes.append(f"{rang} | " + " ".join(f"{c:>{largeur}}" for c in bloc))
    return "\n".join(lignes) + "\n"
