"""
inference.py

Lightweight ONNX-based inference wrapper for the Safety Eye YOLOv8 model.
Replaces ultralytics at runtime — no PyTorch required in production.

Class names match the construction-ppe dataset used during training.
"""

import numpy as np
import cv2
import onnxruntime as ort

CLASS_NAMES = [
    "helmet", "gloves", "vest", "boots", "goggles",
    "none", "Person", "no_helmet", "no_goggle", "no_gloves", "no_boots",
]

# Color palette: green family = compliant, red/orange family = violation
_COLORS = {
    "helmet":    (0, 200, 0),
    "gloves":    (0, 180, 0),
    "vest":      (0, 160, 0),
    "boots":     (0, 140, 0),
    "goggles":   (0, 120, 0),
    "Person":    (255, 165, 0),
    "none":      (0, 0, 255),
    "no_helmet": (0, 0, 230),
    "no_goggle": (30, 0, 220),
    "no_gloves": (60, 0, 210),
    "no_boots":  (90, 0, 200),
}


class SafetyEyeModel:
    """
    ONNX runtime wrapper that mirrors enough of the ultralytics YOLO interface
    to work as a drop-in replacement for inference and annotation.
    """

    def __init__(self, model_path: str, conf_thresh: float = 0.25, iou_thresh: float = 0.45):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.names = {i: name for i, name in enumerate(CLASS_NAMES)}
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, img, verbose: bool = False):
        """
        Run inference on a single image.

        Parameters
        ----------
        img : PIL.Image or numpy ndarray (H, W, 3) in RGB or BGR
        verbose : unused, kept for compatibility

        Returns
        -------
        list[dict]  — each dict has keys:
            class_id, class_name, confidence, bbox (x1, y1, x2, y2)
        """
        arr = self._to_numpy_rgb(img)
        blob, ratio = self._preprocess(arr)
        raw = self.session.run(None, {self.input_name: blob})[0]  # [1, 300, 6]
        return self._postprocess(raw, ratio, arr.shape[:2])

    def plot(self, img, detections):
        """
        Draw bounding boxes on *img* and return an annotated numpy array (BGR).

        Parameters
        ----------
        img : PIL.Image or numpy ndarray (RGB or BGR)
        detections : list[dict] as returned by __call__
        """
        arr = self._to_numpy_rgb(img)
        out = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det["bbox"])
            name = det["class_name"]
            conf = det["confidence"]
            color = _COLORS.get(name, (128, 128, 128))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
            cv2.putText(out, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out  # BGR numpy array

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy_rgb(img):
        """Accept PIL Image or numpy array and return RGB numpy array."""
        if hasattr(img, "convert"):           # PIL Image
            return np.array(img.convert("RGB"))
        if img.ndim == 3 and img.shape[2] == 3:
            # Assume BGR (OpenCV) if it's a numpy array from cv2
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def _preprocess(rgb_arr):
        """
        Letterbox-resize to 640×640, normalize to [0,1], return (blob, scale_ratio).
        """
        h, w = rgb_arr.shape[:2]
        ratio = min(640 / h, 640 / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        resized = cv2.resize(rgb_arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((640, 640, 3), dtype=np.float32)
        canvas[:new_h, :new_w] = resized
        blob = (canvas / 255.0).transpose(2, 0, 1)[None]   # [1, 3, 640, 640]
        return blob.astype(np.float32), ratio

    def _postprocess(self, raw, ratio, orig_hw):
        """
        Decode ONNX output.

        This model exports with NMS baked in, producing shape [1, 300, 6] where
        each row is [x1, y1, x2, y2, confidence, class_id] in 640×640 space.
        Padding rows have confidence == 0.
        """
        preds = raw[0]                           # [300, 6]
        orig_h, orig_w = orig_hw

        results = []
        for row in preds:
            x1, y1, x2, y2, conf, cls = row
            if conf < self.conf_thresh:
                continue
            # Scale from 640×640 letterbox space back to original image
            results.append({
                "class_id":   int(cls),
                "class_name": CLASS_NAMES[int(cls)],
                "confidence": float(conf),
                "bbox": (
                    float(np.clip(x1 / ratio, 0, orig_w)),
                    float(np.clip(y1 / ratio, 0, orig_h)),
                    float(np.clip(x2 / ratio, 0, orig_w)),
                    float(np.clip(y2 / ratio, 0, orig_h)),
                ),
            })
        return results
