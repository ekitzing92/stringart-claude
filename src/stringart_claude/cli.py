"""La commande `stringart-claude`.

stringart-claude creer photo.jpg            # calcule une pièce, écrit les livrables
stringart-claude banc a.jpg b.png           # compare les méthodes sur des images
stringart-claude algos                      # liste les méthodes et celles qu'on recommande
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import banc as banc_essai
from .atelier import creer
from .image import CONTRASTES, charger_image, preparer_cible
from .moteur import Probleme
from .solveurs import RECOMMANDE, SOLVEURS
from .toile import Toile, geometrie


def _options_toile(analyseur: argparse.ArgumentParser) -> None:
    defaut = Toile()
    groupe = analyseur.add_argument_group("la toile")
    groupe.add_argument(
        "--diametre",
        type=float,
        default=defaut.diametre_mm,
        metavar="MM",
        help="diamètre du cercle des clous, en mm (défaut : %(default)g)",
    )
    groupe.add_argument(
        "--clous",
        type=int,
        default=defaut.n_clous,
        metavar="N",
        help="nombre de clous, régulièrement espacés (défaut : %(default)d)",
    )
    groupe.add_argument(
        "--fil",
        type=float,
        default=defaut.epaisseur_fil_mm,
        metavar="MM",
        help="épaisseur du fil, en mm (défaut : %(default)g)",
    )
    groupe.add_argument(
        "--resolution",
        type=int,
        default=defaut.cote,
        metavar="PX",
        help="résolution de travail, en pixels (défaut : %(default)d)",
    )
    groupe = analyseur.add_argument_group("l'image")
    groupe.add_argument(
        "--contraste",
        choices=CONTRASTES,
        default="local",
        help="mise en tons de la cible (défaut : %(default)s)",
    )
    groupe.add_argument(
        "--relief",
        type=float,
        default=0.02,
        help="force du contraste local : 0.01 photo chargée, 0.03 visage (défaut : %(default)g)",
    )
    groupe.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="> 1 assombrit les tons moyens, < 1 les éclaircit (défaut : 1)",
    )
    groupe.add_argument(
        "--noir",
        type=float,
        default=0.3,
        help="valeur visée pour le noir de l'image, de 0 à 1 (défaut : 0.3)",
    )
    groupe.add_argument(
        "--zoom", type=float, default=1.0, help="> 1 resserre le cadrage (défaut : 1)"
    )
    groupe.add_argument("--decalage", type=float, nargs=2, default=(0.0, 0.0),
                        metavar=("DX", "DY"),
                        help="déplace le cadrage (fraction du petit côté)")  # fmt: skip


def _toile(args: argparse.Namespace) -> Toile:
    return Toile(
        diametre_mm=args.diametre, n_clous=args.clous, epaisseur_fil_mm=args.fil,
        cote=args.resolution,
    )  # fmt: skip


def _reglages_image(args: argparse.Namespace) -> dict:
    return {
        "contraste": args.contraste, "relief": args.relief, "gamma": args.gamma,
        "noir": args.noir,
        "zoom": args.zoom, "decalage": tuple(args.decalage),
    }  # fmt: skip


def _texte_algos() -> str:
    lignes = ["Méthodes disponibles (qualité sur 3 étoiles, d'après le banc d'essai) :", ""]
    for s in SOLVEURS.values():
        vitesse = {3: "≈ 1 s", 2: "quelques s", 1: "≈ 10 s"}[s.vitesse]
        marque = "  ← recommandée" if s.recommande else ""
        lignes.append(f"  {s.nom:<19} {s.etoiles}  {vitesse:<11} {s.resume}{marque}")
    return "\n".join(lignes)


def _commande_creer(args: argparse.Namespace) -> int:
    toile = _toile(args)
    print(f"Calcul de {args.image} avec « {args.algo} »…")

    def suivi(etat, n_coups: int) -> None:
        print(f"\r  {etat.n_cordes} cordes, objectif {etat.cout:.5f}", end="", flush=True)

    oeuvre = creer(
        args.image, toile, solveur=args.algo, cordes_max=args.cordes_max,
        suivi=None if args.silencieux else suivi, **_reglages_image(args),
    )  # fmt: skip
    if not args.silencieux:
        print()
    chemins = oeuvre.enregistrer(args.sortie)
    print(oeuvre.resume())
    print(f"Livrables écrits dans {Path(args.sortie).resolve()} :")
    for chemin in chemins.values():
        print(f"  {chemin.name}")
    return 0


def _commande_banc(args: argparse.Namespace) -> int:
    toile = _toile(args)
    geo = geometrie(toile)
    problemes = {
        Path(chemin).stem: Probleme(
            geo, preparer_cible(charger_image(chemin), toile.cote, **_reglages_image(args))
        )
        for chemin in args.images
    }
    noms = args.algos.split(",") if args.algos else None
    mesures, _ = banc_essai.lancer(problemes, noms)
    print(banc_essai.tableau_mesures(mesures))
    print()
    print(banc_essai.tableau_classement(mesures))
    return 0


def construire_analyseur() -> argparse.ArgumentParser:
    analyseur = argparse.ArgumentParser(
        prog="stringart-claude",
        description="String art calculé : de la photo à la séquence de clous.",
    )
    commandes = analyseur.add_subparsers(dest="commande", required=True)

    p = commandes.add_parser(
        "creer", help="calcule une pièce et écrit les livrables",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_texte_algos(),
    )  # fmt: skip
    p.add_argument("image", help="la photo de départ")
    p.add_argument("-o", "--sortie", default="sortie", help="dossier de sortie (défaut : sortie)")
    p.add_argument("--algo", choices=list(SOLVEURS), default=RECOMMANDE,
                   help="méthode de résolution (défaut : %(default)s)")  # fmt: skip
    p.add_argument("--cordes-max", type=int, default=None, metavar="N",
                   help="plafonne le nombre de cordes, donc le temps de montage")  # fmt: skip
    p.add_argument("--silencieux", action="store_true", help="n'affiche pas l'avancement")
    _options_toile(p)
    p.set_defaults(action=_commande_creer)

    p = commandes.add_parser("banc", help="compare les méthodes sur une ou plusieurs images")
    p.add_argument("images", nargs="+", help="les images du banc d'essai")
    p.add_argument("--algos", default=None, metavar="A,B,…",
                   help="méthodes à comparer, séparées par des virgules")  # fmt: skip
    _options_toile(p)
    p.set_defaults(action=_commande_banc)

    p = commandes.add_parser("algos", help="liste les méthodes de résolution")
    p.set_defaults(action=lambda args: print(_texte_algos()) or 0)
    return analyseur


def main(argv: list[str] | None = None) -> int:
    args = construire_analyseur().parse_args(argv)
    try:
        return args.action(args)
    except (ValueError, OSError) as erreur:
        print(f"erreur : {erreur}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
