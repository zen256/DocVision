"""
ocr_engine.py — Tesseract OCR wrapper, возвращает bbox + текст для каждого слова.

Почему Tesseract вместо PaddleOCR:
  - PaddleOCR 'ru'/'cyrillic' путает кириллицу с латиницей (З→S, В→B и т.д.)
  - Tesseract 'rus' обучен на реальных кириллических документах
  - На чётких сканах качество значительно выше

Требования:
  - apt: tesseract-ocr tesseract-ocr-rus
  - pip: pytesseract
"""

from __future__ import annotations

import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Any


# PSM (page segmentation mode) по типу зоны:
#   4 = один столбец текста переменного размера — лучший для таблиц
#   6 = единый блок текста — лучший для заголовков и мета-блоков
_PSM_BY_CLASS = {
    "TITLE":      6,
    "META_BLOCK": 6,
    "TABLE":      4,
}
_PSM_DEFAULT = 6

# Язык: rus+eng — кириллица + латиница + цифры
_LANG = "rus+eng"

# OEM 1 = LSTM (нейросетевой движок, лучше чем legacy OEM 0)
_OEM = 1


class OCREngine:

    def __init__(self, lang: str = _LANG, use_gpu: bool = False):
        """
        lang: язык Tesseract (по умолчанию 'rus+eng')
        use_gpu: игнорируется (Tesseract не использует GPU), оставлен для совместимости
        """
        self._lang = lang
        # Проверяем что tesseract доступен
        try:
            ver = pytesseract.get_tesseract_version()
            print(f"  Tesseract version: {ver}")
        except Exception as e:
            raise RuntimeError(
                f"Tesseract не найден: {e}\n"
                "Установите: sudo apt install tesseract-ocr tesseract-ocr-rus"
            )

    # ── image helpers ──────────────────────────────────────────────────

    def _load_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        return img

    def _crop(self, image: np.ndarray, bbox: list, pad: int = 6) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        return image[max(0, y1 - pad):min(h, y2 + pad),
                     max(0, x1 - pad):min(w, x2 + pad)]

    def _preprocess(self, crop: np.ndarray, zone_class: str = "") -> tuple[np.ndarray, float]:
        """
        Возвращает (обработанный кроп, scale).
        Scale нужен чтобы пересчитать координаты слов обратно в исходные пиксели.
        """
        h, w = crop.shape[:2]

        # Апскейл: Tesseract лучше работает при высоте строки ~40-60px.
        # Типичная строка в нашем документе ~20-24px → апскейл x2.
        scale = 1.0
        if h < 300:   # все наши кропы меньше 300px высоты
            scale = 2.0
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_LANCZOS4)

        return crop, scale

    # ── OCR ────────────────────────────────────────────────────────────

    def run_ocr_words(self, crop: np.ndarray,
                      offset_x: int = 0,
                      offset_y: int = 0,
                      zone_class: str = "") -> List[Dict]:
        """
        Запускает Tesseract на кропе, возвращает список слов:
        [{text, conf, x1, y1, x2, y2}, ...]
        Координаты — в пикселях исходного изображения (с учётом offset и scale).
        """
        processed, scale = self._preprocess(crop, zone_class)

        psm = _PSM_BY_CLASS.get(zone_class.upper(), _PSM_DEFAULT)
        config = f"--oem {_OEM} --psm {psm}"

        # image_to_data возвращает DataFrame с bbox каждого слова
        df = pytesseract.image_to_data(
            processed,
            lang=self._lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        # Символы, которые являются графическими разделителями таблицы,
        # а не текстом. Tesseract часто распознаёт линии таблицы как эти символы.
        _TABLE_NOISE = frozenset('|+─═-_/\\')

        words = []
        n = len(df["text"])
        for i in range(n):
            text = str(df["text"][i]).strip()
            conf = float(df["conf"][i])
            if not text or conf < 0:   # conf=-1 означает не-слово (строка/блок)
                continue

            # Tesseract даёт conf 0-100; отсекаем совсем ненадёжные
            if conf < 20:
                continue

            # Отбрасываем токены состоящие только из символов-разделителей таблицы
            # (вертикальные черты |, горизонтальные линии ─ и т.д.)
            if all(c in _TABLE_NOISE for c in text):
                continue

            # Координаты в масштабированном кропе → исходные пиксели
            x = df["left"][i] / scale + offset_x
            y = df["top"][i]  / scale + offset_y
            bw = df["width"][i]  / scale
            bh = df["height"][i] / scale

            words.append({
                "text": text,
                "conf": conf / 100.0,  # нормализуем в 0..1 как в PaddleOCR
                "x1": x,
                "y1": y,
                "x2": x + bw,
                "y2": y + bh,
            })

        return words

    # ── public: process all YOLO zones ────────────────────────────────

    def process_zones(
        self, image_path: str, zones: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Для каждой зоны запускает OCR и возвращает:
          text  — строки склеенные по строкам (TITLE, META_BLOCK)
          words — список word-dict с координатами (TABLE)
        """
        image    = self._load_image(image_path)
        enriched = []

        for zone in zones:
            x1, y1, x2, y2 = zone["bbox"]
            cls = zone.get("class_name", "")
            crop = self._crop(image, zone["bbox"])

            if crop.size == 0:
                enriched.append({**zone, "text": "", "words": []})
                continue

            try:
                words = self.run_ocr_words(crop,
                                           offset_x=x1,
                                           offset_y=y1,
                                           zone_class=cls)
            except Exception as exc:
                print(f"    ⚠  OCR error [{cls}]: {exc}")
                words = []

            text = words_to_text(words)
            enriched.append({**zone, "text": text, "words": words})

        return enriched


def words_to_text(words: List[Dict]) -> str:
    """Группирует слова в строки по Y-координате и склеивает пробелами."""
    if not words:
        return ""
    sorted_words = sorted(words, key=lambda w: (w["y1"], w["x1"]))
    lines: List[List[Dict]] = []
    current: List[Dict] = [sorted_words[0]]

    for w in sorted_words[1:]:
        prev = current[-1]
        overlap = min(w["y2"], prev["y2"]) - max(w["y1"], prev["y1"])
        h       = max(w["y2"] - w["y1"], prev["y2"] - prev["y1"])
        if overlap > h * 0.3:
            current.append(w)
        else:
            lines.append(current)
            current = [w]
    lines.append(current)

    result_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda w: w["x1"])
        result_lines.append(" ".join(w["text"] for w in line_sorted))
    return "\n".join(result_lines)