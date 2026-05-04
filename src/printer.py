from __future__ import annotations
from typing import Dict, List, Any


def _col_widths(header: List[str], rows: List[Dict]) -> List[int]:
    widths = [max(len(str(h)), 3) for h in header]
    for row in rows:
        for i, h in enumerate(header):
            widths[i] = max(widths[i], len(str(row.get(h, ""))))
    return widths


def _fmt_row(cells: List[str], widths: List[int]) -> str:
    parts = [f" {str(c):<{w}} " for c, w in zip(cells, widths)]
    return "| " + " | ".join(p.strip().ljust(w) for p, w in zip(parts, widths)) + " |"


def _fmt_row2(header: List[str], row: Dict, widths: List[int]) -> str:
    cells = [str(row.get(h, "")) for h in header]
    parts = []
    for c, w in zip(cells, widths):
        parts.append(f" {c:<{w}} ")
    return "|" + "|".join(parts) + "|"


def _sep(widths: List[int], ch: str = "─") -> str:
    return "+" + "+".join(ch * (w + 2) for w in widths) + "+"


class TTNPrinter:

    def print_ttn(self, data: Dict[str, Any]) -> None:
        W = 72
        print("═" * W)

        title = data.get("title", "")
        if title:
            print(f"  {title}")
            print("─" * W)

        meta = data.get("meta", {})
        if meta:
            for k, v in meta.items():
                print(f"  {k}: {v}")
            print("─" * W)

        for t_idx, table in enumerate(data.get("tables", [])):
            header = table.get("header", [])
            rows   = table.get("rows", [])
            if not header:
                continue
            if t_idx > 0:
                print()
            widths = _col_widths(header, rows)
            print(_sep(widths, "─"))
            print(_fmt_row2(header, {h: h for h in header}, widths))
            print(_sep(widths, "═"))
            for row in rows:
                print(_fmt_row2(header, row, widths))
                print(_sep(widths, "─"))

        print("═" * W)