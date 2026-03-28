"""
detector.py — Détection des numéros de roulette dans une vidéo.

Stratégie :
  1. On cherche la zone "NEW >" contenant les badges circulaires des derniers résultats.
  2. On lit le badge le plus à gauche (résultat le plus récent) via OCR.
  3. Quand ce numéro change entre deux lectures, on enregistre un nouveau résultat.
  4. Un délai minimum entre deux résultats évite les doublons pendant l'affichage animé.
"""

import re
import time
import cv2
import numpy as np
import pytesseract
from dataclasses import dataclass, field
from typing import Optional


# ── Configuration ROI (Region Of Interest) ───────────────────────────────────
# Ces valeurs sont des ratios (0.0–1.0) par rapport à la taille de la frame.
# Adaptés au layout du jeu visible dans la capture d'écran.

ROI_PRESETS = {
    # Zone de la bannière "NEW >" avec les badges circulaires de résultats
    "new_badges": {
        "x_start": 0.05,
        "y_start": 0.27,
        "x_end":   0.45,
        "y_end":   0.35,
    },
    # Zone du numéro annoncé en grand au centre (si le jeu affiche un overlay)
    "center_result": {
        "x_start": 0.30,
        "y_start": 0.40,
        "x_end":   0.70,
        "y_end":   0.55,
    },
}

# Numéros valides de roulette européenne (0–36)
VALID_NUMBERS = set(range(37))

# Rouges de la roulette européenne standard
RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 28, 30, 32, 34, 36}


@dataclass
class RouletteResult:
    number:    int
    color:     str          # "red" | "black" | "green"
    parity:    str          # "even" | "odd" | "zero"
    dozen:     str          # "1-12" | "13-24" | "25-36" | "zero"
    timestamp: float        # secondes dans la vidéo
    frame_idx: int


def classify_number(n: int) -> dict:
    """Retourne la couleur, parité et douzaine d'un numéro."""
    if n == 0:
        return {"color": "green", "parity": "zero", "dozen": "zero"}
    color  = "red" if n in RED_NUMBERS else "black"
    parity = "even" if n % 2 == 0 else "odd"
    dozen  = "1-12" if n <= 12 else ("13-24" if n <= 24 else "25-36")
    return {"color": color, "parity": parity, "dozen": dozen}


# ── Prétraitement image ───────────────────────────────────────────────────────

def preprocess_for_ocr(region: np.ndarray) -> np.ndarray:
    """
    Prépare un crop d'image pour Tesseract :
    - Agrandit x3 (Tesseract performe mieux sur les grandes images)
    - Niveaux de gris + seuillage adaptatif
    - Dilation légère pour renforcer les chiffres
    """
    h, w = region.shape[:2]
    region = cv2.resize(region, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    gray   = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Seuillage adaptatif — robuste aux variations d'éclairage
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=4,
    )

    # Dilation pour épaissir les chiffres
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return thresh


def extract_number_from_region(region: np.ndarray) -> Optional[int]:
    """
    Applique OCR sur une région et retourne le premier entier valide (0–36).
    Retourne None si aucun chiffre valide n'est détecté.
    """
    processed = preprocess_for_ocr(region)

    # Config Tesseract : chiffres uniquement, page segmentation mode 7 (ligne unique)
    config = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
    raw    = pytesseract.image_to_string(processed, config=config).strip()

    # On extrait tous les entiers du texte
    numbers = re.findall(r"\d+", raw)
    for tok in numbers:
        n = int(tok)
        if n in VALID_NUMBERS:
            return n
    return None


# ── Détecteur principal ───────────────────────────────────────────────────────

class RouletteDetector:
    """
    Lit une vidéo frame par frame et détecte les nouveaux résultats de roulette.

    Paramètres
    ----------
    video_path       : chemin vers le fichier vidéo
    max_turns        : nombre de tours maximum à capturer (0 = illimité)
    min_gap_seconds  : intervalle minimum entre deux résultats (anti-doublon)
    roi_preset       : "new_badges" ou "center_result"
    debug            : si True, affiche les frames analysées
    """

    def __init__(
        self,
        video_path:      str,
        max_turns:       int   = 0,
        min_gap_seconds: float = 8.0,
        roi_preset:      str   = "new_badges",
        debug:           bool  = False,
    ):
        self.video_path      = video_path
        self.max_turns       = max_turns
        self.min_gap_seconds = min_gap_seconds
        self.roi             = ROI_PRESETS[roi_preset]
        self.debug           = debug

        self._cap            = None
        self._fps            = 30.0
        self._last_number:   Optional[int]  = None
        self._last_ts:       float          = -999.0

    # ── Ouvre / ferme la vidéo ────────────────────────────────────────────────

    def open(self):
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {self.video_path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def close(self):
        if self._cap:
            self._cap.release()
        if self.debug:
            cv2.destroyAllWindows()

    # ── Extraction ROI ────────────────────────────────────────────────────────

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = int(self.roi["x_start"] * w)
        y1 = int(self.roi["y_start"] * h)
        x2 = int(self.roi["x_end"]   * w)
        y2 = int(self.roi["y_end"]   * h)
        return frame[y1:y2, x1:x2]

    # ── Boucle principale ─────────────────────────────────────────────────────

    def run(self) -> list[RouletteResult]:
        """
        Parcourt la vidéo et retourne la liste des résultats détectés.
        Analyse 2 frames par seconde pour réduire la charge CPU.
        """
        self.open()

        results:   list[RouletteResult] = []
        frame_idx: int                  = 0
        # On saute N-1 frames sur N pour analyser ~2 fps
        sample_interval = max(1, int(self._fps / 2))

        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Détecteur] {total_frames} frames | {self._fps:.1f} fps "
              f"| analyse 1 frame / {sample_interval}")

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break

                frame_idx += 1

                # On n'analyse qu'une frame sur sample_interval
                if frame_idx % sample_interval != 0:
                    continue

                timestamp = frame_idx / self._fps
                roi_img   = self._extract_roi(frame)
                number    = extract_number_from_region(roi_img)

                if self.debug:
                    self._show_debug(frame, roi_img, number, timestamp)

                # ── Nouveau résultat ? ────────────────────────────────────────
                if number is not None:
                    gap = timestamp - self._last_ts
                    if number != self._last_number and gap >= self.min_gap_seconds:
                        info   = classify_number(number)
                        result = RouletteResult(
                            number    = number,
                            color     = info["color"],
                            parity    = info["parity"],
                            dozen     = info["dozen"],
                            timestamp = round(timestamp, 2),
                            frame_idx = frame_idx,
                        )
                        results.append(result)
                        self._last_number = number
                        self._last_ts     = timestamp

                        print(f"  ✓ Tour {len(results):>3} | "
                              f"Numéro {number:>2} ({info['color']:<5}) | "
                              f"t={timestamp:.1f}s")

                        if self.max_turns and len(results) >= self.max_turns:
                            print(f"[Détecteur] Limite de {self.max_turns} tours atteinte.")
                            break

        finally:
            self.close()

        print(f"[Détecteur] Terminé — {len(results)} résultats capturés.")
        return results

    # ── Debug ─────────────────────────────────────────────────────────────────

    def _show_debug(self, frame, roi, number, ts):
        # Affiche la ROI avec le numéro détecté
        debug_roi = cv2.resize(roi, (roi.shape[1] * 3, roi.shape[0] * 3))
        label = str(number) if number is not None else "?"
        cv2.putText(debug_roi, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("ROI — Résultat détecté", debug_roi)
        cv2.waitKey(1)
