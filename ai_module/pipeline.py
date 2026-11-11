# -*- coding: utf-8 -*-
"""
Giai đoạn 4 - Ghép nối & xử lý logic.
Điều phối: ảnh vào -> phát hiện vùng biển số (YOLO) -> cắt -> tiền xử lý -> đọc chữ
(PaddleOCR) -> khoanh vùng biển số + từng ký tự -> lưu file -> lưu DB -> trả JSON.

Thứ tự thử (dừng ngay khi ra biển số VN hợp lệ, để nhanh):
  1) Vùng do YOLO tìm ra          - đọc "bền" (có thử lại bằng ảnh phóng to)
  2) Cả ảnh thu nhỏ (cạnh <= 640) - cho ảnh cận cảnh mà YOLO không nhận ra
  3) Vùng do cách dự phòng tìm ra - contour... chỉ đọc 1 lần
  4) Cả ảnh cỡ lớn hơn (cạnh <= 1280)
"""
import os
import time
import cv2

from ai_module import detector, preprocess, ocr_engine, char_segment
from database import db

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINALS_DIR = os.path.join(BASE_DIR, "static", "images", "originals")
CROPS_DIR = os.path.join(BASE_DIR, "static", "images", "crops")

os.makedirs(ORIGINALS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

WHOLE_IMAGE_SMALL_SIDE = 640
WHOLE_IMAGE_LARGE_SIDE = 1280


def _imwrite_unicode(path, image):
    """Thay cv2.imwrite() vì nó ghi thất bại âm thầm với đường dẫn có dấu tiếng Việt."""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"Không encode được ảnh trước khi lưu: {path}")
    buffer.tofile(path)


# ---------------------------------------------------------------------------
# Đọc chữ
# ---------------------------------------------------------------------------
def _read_plate(image, robust):
    """
    Trả về (text, conf, hop_cac_dong). robust=True: nếu kết quả bất thường thì thử lại bằng
    ảnh phóng to; robust=False: đọc đúng 1 lần (cho vùng kém tin cậy, đỡ mất thời gian).
    Hộp các dòng theo toạ độ của chính `image` truyền vào.
    """
    if robust:
        return ocr_engine.read_plate_robust(image, return_lines=True)
    return ocr_engine.read_plate_paddleocr(image, return_lines=True)


def _result_score(text, conf):
    """Xếp hạng kết quả: biển VN hợp lệ > đọc được từ 6 ký tự > độ tin cậy."""
    n_alnum = sum(ch.isalnum() for ch in text)
    return (ocr_engine.is_valid_vn_plate(text), n_alnum >= 6, conf)


def _abs_box(x, y, w, h):
    return {"x": int(round(x)), "y": int(round(y)), "w": max(1, int(round(w))), "h": max(1, int(round(h)))}


def _union_box(boxes, img_w, img_h, pad_ratio=0.10):
    """Hộp bao quanh các hộp dòng chữ, nới ra một chút, cắt theo khung ảnh."""
    x0 = min(b["x"] for b in boxes)
    y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["w"] for b in boxes)
    y1 = max(b["y"] + b["h"] for b in boxes)
    pad_x = max(6, int((x1 - x0) * pad_ratio))
    pad_y = max(6, int((y1 - y0) * pad_ratio))
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(img_w, x1 + pad_x), min(img_h, y1 + pad_y)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _attempt_box(image_bgr, box, robust):
    """Đọc trong 1 vùng ứng viên. Trả về dict kết quả (hộp dòng theo toạ độ ẢNH GỐC) hoặc None."""
    x, y = max(0, box["x"]), max(0, box["y"])
    crop = image_bgr[y:y + box["h"], x:x + box["w"]]
    if crop.size == 0:
        return None

    display_crop, _binary_unused = preprocess.enhance_for_ocr(crop)
    t1 = time.time()
    text, conf, line_boxes = _read_plate(display_crop, robust)

    # display_crop có thể đã phóng to so với crop -> quy đổi hộp dòng về toạ độ ảnh gốc
    ch, cw = crop.shape[:2]
    sx, sy = cw / float(display_crop.shape[1]), ch / float(display_crop.shape[0])
    lines_abs = [_abs_box(x + b["x"] * sx, y + b["y"] * sy, b["w"] * sx, b["h"] * sy) for b in line_boxes]

    print(f"[pipeline] vung {box.get('source', '?')} {box['w']}x{box['h']}: '{text}' ({time.time() - t1:.1f}s)")
    return {"box": {**box, "x": x, "y": y}, "display": display_crop, "text": text, "conf": conf,
            "lines": lines_abs, "score": _result_score(text, conf)}


def _attempt_whole(image_bgr, max_side):
    """
    Đọc cả ảnh (thu nhỏ về cạnh dài <= max_side cho nhanh). Hộp biển số là hộp bao quanh
    các dòng chữ thuộc biển số, không phải cả ảnh.
    """
    H, W = image_bgr.shape[:2]
    scale = min(1.0, max_side / float(max(H, W)))
    whole = image_bgr if scale >= 1.0 else cv2.resize(
        image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    t1 = time.time()
    text, conf, line_boxes = _read_plate(whole, robust=False)
    print(f"[pipeline] ca anh (canh <= {max_side}): "
          f"{repr(text) if text else 'khong thay gi giong bien so'} ({time.time() - t1:.1f}s)")

    if not text:
        # Không có box/display: ứng viên này sẽ không bao giờ được vẽ lên ảnh hay hiển thị
        # như một kết quả tìm thấy (process_frame chỉ vẽ/lưu khi box is not None). Vẫn giữ
        # `conf` (0.0 hoặc -1.0, xem read_plate_paddleocr) để _pick_and_read_best biết có
        # nên thử đọc lại ở độ phân giải lớn hơn hay không.
        return {"box": None, "display": None, "text": "", "conf": conf, "lines": [],
                "score": _result_score("", conf)}

    lines_abs = [_abs_box(b["x"] / scale, b["y"] / scale, b["w"] / scale, b["h"] / scale) for b in line_boxes]

    box = _union_box(lines_abs, W, H) if lines_abs else {"x": 0, "y": 0, "w": W, "h": H}
    box.update(score=0.01, source="whole_image")
    display = image_bgr[box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]]

    return {"box": box, "display": display, "text": text, "conf": conf,
            "lines": lines_abs, "score": _result_score(text, conf)}


def _pick_and_read_best(image_bgr, boxes):
    """
    Trả về (hop_bien_so, anh_crop, chu, do_tin_cay, hop_cac_dong_chu) — toạ độ theo ảnh gốc.

    Để nhanh: đọc "bền" (phóng to nhiều lần) rất tốn thời gian, nên chỉ dùng ĐÚNG 1 LẦN
    trên ứng viên tốt nhất, sau khi đọc nhanh (không phóng to) tất cả ứng viên mà chưa ra
    kết quả hợp lệ. Dừng ngay lập tức ở bất kỳ bước nào ngay khi có biển số VN hợp lệ,
    không thử tiếp các ứng viên/bước còn lại.
    """
    if not boxes:
        return None, None, "", 0.0, []

    H, W = image_bgr.shape[:2]

    def better(a, b):
        return a is not None and (b is None or a["score"] > b["score"])

    def is_ok(a):
        return a is not None and ocr_engine.is_valid_vn_plate(a["text"])

    ordered = sorted(boxes, key=lambda b: (b["score"], b["w"] * b["h"]), reverse=True)
    yolo_boxes = [b for b in ordered if b.get("source") == "yolov8"][:3]
    # Vùng dự phòng to gần bằng cả ảnh thì đã được bước "đọc cả ảnh" bao trùm -> bỏ qua
    other_boxes = [b for b in ordered
                   if b.get("source") != "yolov8" and b["w"] * b["h"] < 0.6 * W * H][:2]

    best = None

    # Bước 1: đọc NHANH (1 lần, không phóng to) từng vùng YOLO; dừng ngay khi có kết quả hợp lệ
    for box in yolo_boxes:
        a = _attempt_box(image_bgr, box, robust=False)
        if better(a, best):
            best = a
        if is_ok(best):
            return best["box"], best["display"], best["text"], best["conf"], best["lines"]

    # Bước 2: chưa hợp lệ -> thử đọc BỀN (phóng to) DUY NHẤT trên vùng YOLO tốt nhất
    if yolo_boxes:
        a = _attempt_box(image_bgr, yolo_boxes[0], robust=True)
        if better(a, best):
            best = a
        if is_ok(best):
            return best["box"], best["display"], best["text"], best["conf"], best["lines"]

    # Bước 3: đọc cả ảnh thu nhỏ (cho ảnh cận cảnh mà YOLO không nhận ra)
    whole_small = _attempt_whole(image_bgr, WHOLE_IMAGE_SMALL_SIDE)
    if better(whole_small, best):
        best = whole_small
    if is_ok(best):
        return best["box"], best["display"], best["text"], best["conf"], best["lines"]

    # Bước 4: các vùng dự phòng (contour...), đọc nhanh, dừng sớm nếu hợp lệ
    for box in other_boxes:
        a = _attempt_box(image_bgr, box, robust=False)
        if better(a, best):
            best = a
        if is_ok(best):
            return best["box"], best["display"], best["text"], best["conf"], best["lines"]

    # Bước 5: cả ảnh ở kích thước lớn hơn — bỏ qua CHỈ KHI bước 3 đã xác định rõ đây không
    # phải biển số (conf=-1.0: đã thấy chữ nhưng không dòng nào giống biển số, kiểu ảnh sản
    # phẩm/nhãn/nhiều chữ khác — đọc lớn hơn cũng vô ích). Dùng kết quả BƯỚC 3 riêng
    # (whole_small), không dùng "best" hiện tại, vì "best" có thể đã bị bước 4 (contour trả
    # về rỗng với conf=0.0 bình thường) ghi đè mất tín hiệu -1.0 này.
    definitely_not_plate = whole_small is not None and whole_small["conf"] < 0
    if not is_ok(best) and max(H, W) > WHOLE_IMAGE_SMALL_SIDE and not definitely_not_plate:
        a = _attempt_whole(image_bgr, WHOLE_IMAGE_LARGE_SIDE)
        if better(a, best):
            best = a

    if best is None:
        return None, None, "", 0.0, []
    return best["box"], best["display"], best["text"], best["conf"], best["lines"]


# ---------------------------------------------------------------------------
# Xử lý 1 khung hình / 1 ảnh upload
# ---------------------------------------------------------------------------
def process_frame(image_bgr):
    """Input: ảnh BGR. Output: dict kết quả đầy đủ, đã lưu ảnh + ghi DB."""
    t0 = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"

    original_filename = f"goc_{timestamp}.jpg"
    original_path_abs = os.path.join(ORIGINALS_DIR, original_filename)
    _imwrite_unicode(original_path_abs, image_bgr)
    original_url = f"/static/images/originals/{original_filename}"

    boxes = detector.detect(image_bgr)
    best_box, display_crop, plate_text, confidence, line_boxes = _pick_and_read_best(image_bgr, boxes)
    confidence = max(confidence, 0.0)

    # Ảnh bước 2 — CHỈ khung vùng biển số do bước phát hiện tìm ra, chưa có nhãn/ký tự.
    # Lưu NGAY ở đây (trước khi biết chữ đọc được) để đúng tinh thần "ảnh của riêng bước
    # phát hiện", tách bạch với bước đọc chữ ở sau — phục vụ minh hoạ từng bước trong
    # mục Lịch sử > Xem chi tiết.
    det_image = image_bgr.copy()
    if best_box is not None:
        dx, dy = max(0, best_box["x"]), max(0, best_box["y"])
        dw, dh = best_box["w"], best_box["h"]
        d_scale = max(1.0, max(det_image.shape[:2]) / 800.0)
        cv2.rectangle(det_image, (dx, dy), (dx + dw, dy + dh), (0, 200, 255), max(2, int(round(3 * d_scale))))
    det_filename = f"det_{timestamp}.jpg"
    _imwrite_unicode(os.path.join(ORIGINALS_DIR, det_filename), det_image)

    # Kết quả quá ngắn (1-3 ký tự) mà cũng không phải biển số VN hợp lệ: gần như chắc chắn là
    # chữ vụn đọc nhầm từ ảnh không phải biển số (nhãn sản phẩm, chữ trang trí...), không phải
    # biển số thật đọc thiếu. Coi như không phát hiện được gì, không hiển thị như thể là biển số.
    if plate_text and not ocr_engine.is_valid_vn_plate(plate_text) and ocr_engine._alnum_len(plate_text) < 4:
        best_box, display_crop, plate_text, confidence, line_boxes = None, None, "", 0.0, []

    output_image = image_bgr.copy()
    crop_url = None
    box_payload = None
    char_boxes = []

    if best_box is not None and display_crop is not None:
        img_h, img_w = image_bgr.shape[:2]
        x, y = max(0, best_box["x"]), max(0, best_box["y"])
        w, h = min(best_box["w"], img_w - x), min(best_box["h"], img_h - y)

        crop_filename = f"crop_{timestamp}.jpg"
        crop_path_abs = os.path.join(CROPS_DIR, crop_filename)
        _imwrite_unicode(crop_path_abs, display_crop)
        crop_url = f"/static/images/crops/{crop_filename}"

        # Nét vẽ / chữ tự phóng theo cỡ ảnh, để ảnh lớn (vài nghìn px) vẫn thấy rõ
        draw_scale = max(1.0, max(img_h, img_w) / 800.0)
        box_thick = max(2, int(round(3 * draw_scale)))
        text_thick = max(2, int(round(2 * draw_scale)))
        font_scale = 0.9 * min(draw_scale, 3.5)

        color_plate = (0, 200, 255)   # vàng hổ phách (BGR)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color_plate, box_thick)

        label = plate_text if plate_text else "Khong doc duoc"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick)
        pad = int(14 * min(draw_scale, 3.5))
        if y - th - pad >= 0:
            label_top, label_bottom, text_y = y - th - pad, y, y - pad // 2
        else:
            # Khung sát mép trên ảnh: đặt nhãn bên trong khung để không bị cắt mất
            label_top, label_bottom, text_y = y, y + th + pad, y + th + pad // 2
        cv2.rectangle(output_image, (x, label_top), (x + tw + pad, label_bottom), color_plate, -1)
        cv2.putText(output_image, label, (x + pad // 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), text_thick, cv2.LINE_AA)

        # Khoanh từng ký tự bên trong các dòng chữ của biển số
        color_char = (60, 220, 60)
        char_thick = max(1, int(round(1.5 * draw_scale)))
        char_boxes = char_segment.characters_in_lines(image_bgr, line_boxes)
        for cb in char_boxes:
            cv2.rectangle(output_image, (cb["x"], cb["y"]),
                          (cb["x"] + cb["w"], cb["y"] + cb["h"]), color_char, char_thick)

        # Ảnh bước "tách ký tự" riêng (không dính khung/nhãn to của bước cuối, dễ nhìn khi
        # phóng lớn): vẽ khung ký tự lên 1 bản sao sạch rồi cắt đúng vùng biển số ra.
        char_only = image_bgr.copy()
        for cb in char_boxes:
            cv2.rectangle(char_only, (cb["x"], cb["y"]),
                          (cb["x"] + cb["w"], cb["y"] + cb["h"]), color_char, char_thick)
        char_crop = char_only[y:y + h, x:x + w]
        if char_crop.size > 0:
            char_filename = f"char_{timestamp}.jpg"
            _imwrite_unicode(os.path.join(CROPS_DIR, char_filename), char_crop)

        box_payload = {"x": x, "y": y, "w": w, "h": h}

    output_filename = f"out_{timestamp}.jpg"
    output_path_abs = os.path.join(ORIGINALS_DIR, output_filename)
    _imwrite_unicode(output_path_abs, output_image)
    output_url = f"/static/images/originals/{output_filename}"

    record_id = db.insert_lich_su(
        bien_so_text=plate_text or None,
        do_tin_cay=confidence,
        duong_dan_anh_goc=original_url,
        duong_dan_anh_crop=crop_url,
    )

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "id": record_id,
        "bien_so": plate_text,
        "do_tin_cay": confidence,
        "anh_dau_vao": original_url,
        "anh_dau_ra": output_url,
        "anh_crop": crop_url,
        "khung": box_payload,
        "so_ky_tu_tach_duoc": len(char_boxes),
        "phat_hien_duoc": best_box is not None,
        "thoi_gian_xu_ly_ms": elapsed_ms,
    }