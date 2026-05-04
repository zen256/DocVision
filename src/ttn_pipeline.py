import sys
import json
import argparse
from pathlib import Path

from detector import YOLODetector
from ocr_engine import OCREngine
from parser import StructuralParser
from printer import TTNPrinter


def run_pipeline(image_path: str, model_path: str, output_json: str = None, conf: float = 0.3) -> dict:
    print(f"\n{'='*60}")
    print("  ТТН Recognition Pipeline")
    print(f"{'='*60}")
    print(f"  Изображение : {image_path}")
    print(f"  Модель      : {model_path}")
    print(f"{'='*60}\n")

    print("[1/3] Запуск YOLO-детектора...")
    detector = YOLODetector(model_path, conf_threshold=conf)
    zones = detector.detect(image_path)

    if not zones:
        print("YOLO не нашёл ни одной зоны.")
        return {}

    print(f"Найдено зон: {len(zones)}")
    for z in zones:
        print(f"    [{z['class_name']:12s}] conf={z['conf']:.2f}  bbox={z['bbox']}")

    print("\n[2/3] OCR в каждой зоне...")
    ocr = OCREngine()
    zones_with_text = ocr.process_zones(image_path, zones)

    for z in zones_with_text:
        lines = z.get("text", "").count("\n") + 1
        print(f"  [{z['class_name']:12s}] → {lines} строк(и) текста")

    print("\n[3/3]  Структурный парсер...")
    parser = StructuralParser()
    result = parser.parse(zones_with_text)

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nJSON сохранён: {out_path}")

    print()
    printer = TTNPrinter()
    printer.print_ttn(result)

    return result


def main():
    parser = argparse.ArgumentParser(description="TTN Recognition Pipeline")
    parser.add_argument("image", help="Путь к изображению ТТН")
    parser.add_argument(
        "--model",
        default="./models/yolo_ttn_model.pt",
        help="Путь к весам YOLO (best.pt)",
    )
    parser.add_argument("--output", default="output.json", help="Путь для сохранения JSON")
    parser.add_argument("--conf", type=float, default=0.3, help="Порог уверенности YOLO (0-1)")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Изображение не найдено: {args.image}")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"Модель не найдена: {args.model}")
        sys.exit(1)

    run_pipeline(args.image, args.model, args.output, args.conf)

if __name__ == "__main__":
    main()
