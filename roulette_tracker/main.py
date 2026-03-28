#!/usr/bin/env python3
"""
main.py — Point d'entrée du programme de tracking de roulette.

Usage
-----
  # Analyser une vidéo, capturer 50 tours max
  python main.py video.mp4 --turns 50

  # Analyser jusqu'à la fin, avec debug visuel
  python main.py video.mp4 --debug

  # Charger un CSV existant et regénérer les stats
  python main.py --load results/results.csv

  # Calibrer manuellement la zone de détection
  python main.py video.mp4 --calibrate
"""

import argparse
import sys
from pathlib import Path

from detector import RouletteDetector
from storage  import save_all, load_csv
from stats    import generate_all


# ── Calibration interactive ───────────────────────────────────────────────────

def calibrate(video_path: str):
    """
    Affiche la première frame de la vidéo et permet de sélectionner
    la ROI manuellement via selectROI d'OpenCV.
    Affiche les ratios à copier dans detector.py.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Impossible de lire la vidéo.")
        return

    h, w = frame.shape[:2]
    print(f"Résolution vidéo : {w}×{h}")
    print("Sélectionnez la zone contenant les derniers résultats (boutons NEW),")
    print("puis appuyez sur ESPACE ou ENTRÉE pour valider.")

    roi = cv2.selectROI("Calibration — sélectionnez la zone NEW", frame, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, rw, rh = roi
    print("\n── Ratios à copier dans detector.py ──────────────────────────")
    print(f'    "x_start": {x/w:.3f},')
    print(f'    "y_start": {y/h:.3f},')
    print(f'    "x_end":   {(x+rw)/w:.3f},')
    print(f'    "y_end":   {(y+rh)/h:.3f},')
    print("──────────────────────────────────────────────────────────────")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tracker de résultats de roulette depuis une vidéo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("video", nargs="?",
                   help="Chemin vers la vidéo à analyser")
    p.add_argument("--turns", "-n", type=int, default=0,
                   help="Nombre maximum de tours à capturer (0 = illimité)")
    p.add_argument("--gap", "-g", type=float, default=8.0,
                   help="Délai minimum (secondes) entre deux résultats (défaut : 8)")
    p.add_argument("--roi", choices=["new_badges", "center_result"],
                   default="new_badges",
                   help="Zone de détection (défaut : new_badges)")
    p.add_argument("--output", "-o", default="results",
                   help="Dossier de sortie pour CSV/JSON/graphiques (défaut : results/)")
    p.add_argument("--name", default="results",
                   help="Nom de base pour les fichiers de sortie (défaut : results)")
    p.add_argument("--debug", action="store_true",
                   help="Affiche la ROI analysée en temps réel")
    p.add_argument("--load",
                   help="Charge un CSV existant et regénère les statistiques")
    p.add_argument("--calibrate", action="store_true",
                   help="Lance l'outil de calibration de la ROI")
    p.add_argument("--no-stats", action="store_true",
                   help="Ne génère pas les graphiques")
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    # ── Mode : charger un CSV existant ───────────────────────────────────────
    if args.load:
        print(f"Chargement de {args.load} …")
        results = load_csv(args.load)
        print(f"{len(results)} résultats chargés.")
        if not args.no_stats:
            generate_all(results, args.output)
        return

    # ── Mode : calibration ────────────────────────────────────────────────────
    if args.calibrate:
        if not args.video:
            print("Erreur : --calibrate nécessite un fichier vidéo.")
            sys.exit(1)
        calibrate(args.video)
        return

    # ── Mode : analyse vidéo ──────────────────────────────────────────────────
    if not args.video:
        build_parser().print_help()
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Erreur : fichier introuvable — {video_path}")
        sys.exit(1)

    print(f"\n{'═'*55}")
    print(f"  BirdCLEF Roulette Tracker")
    print(f"{'═'*55}")
    print(f"  Vidéo       : {video_path}")
    print(f"  Tours max   : {args.turns or 'illimité'}")
    print(f"  Gap min     : {args.gap}s")
    print(f"  ROI         : {args.roi}")
    print(f"  Sortie      : {args.output}/")
    print(f"{'═'*55}\n")

    # Détection
    detector = RouletteDetector(
        video_path      = str(video_path),
        max_turns       = args.turns,
        min_gap_seconds = args.gap,
        roi_preset      = args.roi,
        debug           = args.debug,
    )
    results = detector.run()

    if not results:
        print("Aucun résultat détecté. Vérifiez la ROI avec --calibrate.")
        sys.exit(0)

    # Sauvegarde
    paths = save_all(results, args.output, args.name)
    print(f"\nFichiers créés :")
    for fmt, p in paths.items():
        print(f"  {fmt.upper():5} → {p}")

    # Statistiques
    if not args.no_stats:
        generate_all(results, args.output)

    print(f"\nTerminé — {len(results)} tours enregistrés.\n")


if __name__ == "__main__":
    main()
