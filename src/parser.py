from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from ocr_engine import words_to_text


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _split_rows(text: str) -> List[str]:
    return [_clean(l) for l in text.splitlines() if _clean(l)]


#META парсер

_META_RE = re.compile(r"^(?P<key>[^:：]{2,40}?)\s*[:：]\s*(?P<value>.+)$")

def parse_meta_block(text: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for line in _split_rows(text):
        m = _META_RE.match(line)
        if m:
            meta[_clean(m.group("key"))] = _clean(m.group("value"))
    return meta


#TABLE парсер через bbox

def _cluster_1d(values: List[float], gap: float) -> List[List[int]]:
    """
    Кластеризует 1D значения (координаты) с порогом gap.
    Возвращает список кластеров — каждый кластер это список индексов.
    """
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    clusters: List[List[int]] = [[indexed[0][0]]]
    for i in range(1, len(indexed)):
        if indexed[i][1] - indexed[i-1][1] > gap:
            clusters.append([])
        clusters[-1].append(indexed[i][0])
    return clusters


def parse_table_from_words(words: List[Dict]) -> Dict[str, Any]:
    if not words:
        return {"header": [], "rows": []}

    #Группировка в строки по Y
    sorted_w = sorted(words, key=lambda w: w["y1"])
    row_groups: List[List[Dict]] = []
    current = [sorted_w[0]]

    for w in sorted_w[1:]:
        prev = current[-1]
        overlap = min(w["y2"], prev["y2"]) - max(w["y1"], prev["y1"])
        h = max(w["y2"] - w["y1"], prev["y2"] - prev["y1"], 1)
        if overlap > h * 0.25:
            current.append(w)
        else:
            row_groups.append(current)
            current = [w]
    row_groups.append(current)

    if len(row_groups) < 2:
        # Fallback: вся таблица как одна строка
        text = " ".join(w["text"] for w in words)
        return {"header": [text], "rows": []}

    #Находим границы колонок по заголовку
    # Первая строка = заголовок, определяем x-центры колонок
    header_words = sorted(row_groups[0], key=lambda w: w["x1"])
    col_centers = [(w["x1"] + w["x2"]) / 2 for w in header_words]
    header_texts = [w["text"] for w in header_words]

    n_cols = len(header_texts)
    if n_cols == 0:
        return {"header": [], "rows": []}

    #Границы колонок: середина между соседними центрами
    col_bounds = [0.0]
    for i in range(1, n_cols):
        col_bounds.append((col_centers[i-1] + col_centers[i]) / 2)
    col_bounds.append(float("inf"))

    def assign_col(x_center: float) -> int:
        for i in range(n_cols):
            if col_bounds[i] <= x_center < col_bounds[i+1]:
                return i
        return n_cols - 1

    #Строим строки
    rows: List[Dict[str, str]] = []
    for row_words in row_groups[1:]:
        cells = [""] * n_cols
        for w in sorted(row_words, key=lambda w: w["x1"]):
            cx = (w["x1"] + w["x2"]) / 2
            col = assign_col(cx)
            cells[col] = (cells[col] + " " + w["text"]).strip()
        record = {header_texts[i]: cells[i] for i in range(n_cols)}
        rows.append(record)

    return {"header": header_texts, "rows": rows}


#Главный парсер 
class StructuralParser:

    def parse(self, zones: List[Dict[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "title" : "",
            "meta"  : {},
            "tables": [],
            "_zones": [],
        }
        table_idx = 0

        for zone in zones:
            cls   = zone["class_name"].upper()
            text  = zone.get("text", "")
            words = zone.get("words", [])
            bbox  = zone.get("bbox", [])
            conf  = zone.get("conf", 0.0)

            result["_zones"].append({
                "class": cls, "conf": round(conf, 3),
                "bbox": bbox, "text": text,
            })

            if cls == "TITLE":
                lines = _split_rows(text)
                if lines:
                    result["title"] = lines[0]

            elif cls == "META_BLOCK":
                result["meta"].update(parse_meta_block(text))

            elif cls == "TABLE":
                if words:
                    tbl = parse_table_from_words(words)
                else:
                    # Fallback: парсим из текста
                    tbl = _parse_table_from_text(text)

                result["tables"].append({
                    "table_index": table_idx,
                    "header"     : tbl["header"],
                    "rows"       : tbl["rows"],
                })
                table_idx += 1

        return result


def _parse_table_from_text(text: str) -> Dict[str, Any]:
    """Fallback: разбор таблицы из plain text."""
    rows = _split_rows(text)
    if len(rows) < 2:
        return {"header": rows, "rows": []}

    header = re.split(r"\s{2,}", rows[0])
    data_rows = []
    for row in rows[1:]:
        cells = re.split(r"\s{2,}", row)
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        data_rows.append({header[i]: cells[i] for i in range(len(header))})
    return {"header": header, "rows": data_rows}