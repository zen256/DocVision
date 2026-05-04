import json
import tempfile
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory

# Импорт пайплайна
from ttn_pipeline import run_pipeline

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Путь к модели по умолчанию
DEFAULT_MODEL = "./models/yolo_ttn_model.pt"


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return jsonify({"error": "Файл не передан"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Имя файла пустое"}), 400

    model_path = request.form.get("model_path", DEFAULT_MODEL)
    conf       = float(request.form.get("conf", 0.3))

    # Сохраняем загруженный файл во временную директорию
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = run_pipeline(
            image_path=tmp_path,
            model_path=model_path,
            output_json=None,
            conf=conf,
        )
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"ok": False, "error": str(e), "traceback": tb}), 500
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
