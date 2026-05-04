from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

CLASS_NAMES = {
    0: "TITLE",
    1: "META_BLOCK",
    2: "TABLE",
}

class YOLODetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.3):
        from ultralytics import YOLO

        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)

        if hasattr(self.model, "names") and self.model.names:
            self._class_names = self.model.names 
        else:
            self._class_names = CLASS_NAMES

    def detect(self, image_path: str) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            verbose=False,
        )

        zones: List[Dict[str, Any]] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf   = float(boxes.conf[i].item())
                xyxy   = boxes.xyxy[i].cpu().numpy().astype(int).tolist()  # [x1,y1,x2,y2]

                cls_name = self._class_names.get(cls_id, f"CLASS_{cls_id}")

                zones.append(
                    {
                        "class_id"  : cls_id,
                        "class_name": cls_name,
                        "conf"      : conf,
                        "bbox"      : xyxy,
                    }
                )

        zones.sort(key=lambda z: z["bbox"][1])
        return zones

    def crop_zone(self, image: np.ndarray, bbox: list, padding: int = 4) -> np.ndarray:
        """
        Crop a zone from a numpy image with optional padding.
        Safe against out-of-bounds.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        return image[y1:y2, x1:x2]
