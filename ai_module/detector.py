# -*- coding: utf-8 -*-
"""
Giai đoạn 2 - Phát hiện (Detection).
Ưu tiên: YOLOv8 (ảnh gốc) > YOLOv8 (ảnh có đệm viền, cho ảnh cận cảnh)
> Haar Cascade > contour cổ điển (Top Hat/Black Hat) > vùng trung tâm ảnh.
"""
import os
import cv2
import numpy as np

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_YOLO_WEIGHTS = os.path.join(_BASE_DIR, "models", "yolov8_plate.pt")
_HAAR_XML_LOCAL = os.path.join(_BASE_DIR, "models", "haarcascade_russian_plate_number.xml")


# ---------------------------------------------------------------------------
# YOLOv8
# ---------------------------------------------------------------------------
_yolo_model = None
_yolo_checked = False


def _try_load_yolo():
    global _yolo_model, _yolo_checked
    if _yolo_checked:
        return _yolo_model
    _yolo_checked = True
    if os.path.exists(_YOLO_WEIGHTS):
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO(_YOLO_WEIGHTS)
        except Exception as e:
            print(f"[detector] Không load được YOLO ({e}); dùng Haar/contour thay thế.")
            _yolo_model = None
    return _yolo_model


def _detect_yolo(image_bgr, model):
    boxes = []
    results = model.predict(image_bgr, conf=0.15, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "score": conf, "source": "yolov8"})
    return boxes


def _detect_yolo_with_padding(image_bgr, model, pad_ratio=0.35):
    """
    Thêm viền đệm màu xám quanh ảnh trước khi đưa vào YOLO — dùng cho ảnh
    cận cảnh (biển số chiếm gần hết khung hình), mô phỏng lại bối cảnh
    giống đa số ảnh huấn luyện (biển số chỉ chiếm 1 phần nhỏ khung hình).
    """
    h, w = image_bgr.shape[:2]
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
    padded = cv2.copyMakeBorder(
        image_bgr, pad_y, pad_y, pad_x, pad_x,
        cv2.BORDER_CONSTANT, value=(90, 90, 90)
    )
    raw_boxes = _detect_yolo(padded, model)
    boxes = []
    for b in raw_boxes:
        boxes.append({**b, "x": b["x"] - pad_x, "y": b["y"] - pad_y})
    return boxes



# ---------------------------------------------------------------------------
# Nới rộng khung YOLO một chút trước khi cắt. Khung YOLO có thể hụt vài % chiều rộng (nhất là
# ảnh cận cảnh, khác phong cách ảnh huấn luyện) -> ký tự sát mép (đầu/cuối biển số) bị cắt đứt
# ngay từ bước phát hiện, OCR không thể đọc lại thứ đã bị cắt mất (vd '036.58' thành '036.5').
# Chữ nằm trong vùng đệm thừa thì vô hại: PaddleOCR tự tìm dòng chữ trong crop.
# ---------------------------------------------------------------------------
YOLO_BOX_PAD_RATIO = 0.12


def _expand_boxes(boxes, img_w, img_h, ratio=YOLO_BOX_PAD_RATIO):
    out = []
    for b in boxes:
        px, py = int(b["w"] * ratio), int(b["h"] * ratio)
        x0 = max(0, b["x"] - px)
        y0 = max(0, b["y"] - py)
        x1 = min(img_w, b["x"] + b["w"] + px)
        y1 = min(img_h, b["y"] + b["h"] + py)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        out.append({**b, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    return out

# ---------------------------------------------------------------------------
# Haar Cascade (tuỳ chọn, nếu có file .xml)
# ---------------------------------------------------------------------------
_haar_cascade = None
_haar_checked = False


def _try_load_haar():
    global _haar_cascade, _haar_checked
    if _haar_checked:
        return _haar_cascade
    _haar_checked = True

    candidate_paths = [_HAAR_XML_LOCAL]
    try:
        candidate_paths.append(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    except AttributeError:
        pass

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                cascade = cv2.CascadeClassifier(path)
            except AttributeError:
                break
            if not cascade.empty():
                _haar_cascade = cascade
                return _haar_cascade

    print("[detector] Không tìm thấy Haar Cascade hợp lệ; dùng contour thay thế.")
    _haar_cascade = None
    return _haar_cascade


def _detect_haar(image_bgr):
    cascade = _try_load_haar()
    if cascade is None:
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    plates = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(60, 20))
    boxes = []
    for (x, y, w, h) in plates:
        boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "score": 0.75, "source": "haar"})
    return boxes


# ---------------------------------------------------------------------------
# Contour cổ điển (Top Hat / Black Hat chống chói/phản sáng)
# ---------------------------------------------------------------------------
def _detect_contour(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.add(gray, top_hat)
    enhanced = cv2.subtract(enhanced, black_hat)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 200)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    img_h, img_w = gray.shape[:2]
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w < 40 or h < 12:
            continue
        aspect = w / float(h)
        area_ratio = (w * h) / float(img_w * img_h)
        looks_like_plate = (1.5 <= aspect <= 5.5) or (0.85 <= aspect <= 1.8)
        if looks_like_plate and 0.002 <= area_ratio <= 0.6:
            boxes.append({"x": x, "y": y, "w": w, "h": h, "score": 0.5, "source": "contour_tophat"})
    return boxes[:3]


# ---------------------------------------------------------------------------
# Hàm chính
# ---------------------------------------------------------------------------
def detect(image_bgr):
    """
    Trả về danh sách box ứng viên biển số: [{x, y, w, h, score, source}, ...]
    """
    model = _try_load_yolo()
    img_h, img_w = image_bgr.shape[:2]
    if model is not None:
        boxes = _detect_yolo(image_bgr, model)
        if boxes:
            return _expand_boxes(boxes, img_w, img_h)

        boxes = _detect_yolo_with_padding(image_bgr, model)
        if boxes:
            return _expand_boxes(boxes, img_w, img_h)

    boxes = _detect_haar(image_bgr)
    if boxes:
        return boxes

    boxes = _detect_contour(image_bgr)
    if boxes:
        return boxes

    # Không phương án nào tìm thấy gì -> thử vùng trung tâm ảnh (bỏ viền 12%),
    # phòng trường hợp ảnh cận cảnh chỉ có mỗi biển số, dính chút nhiễu ở rìa.
    h, w = image_bgr.shape[:2]
    margin_x, margin_y = int(w * 0.12), int(h * 0.12)
    print("[detector] Khong detector nao tim thay vung; thu vung trung tam anh.")
    return [{
        "x": margin_x, "y": margin_y,
        "w": w - 2 * margin_x, "h": h - 2 * margin_y,
        "score": 0.05, "source": "center_fallback"
    }]