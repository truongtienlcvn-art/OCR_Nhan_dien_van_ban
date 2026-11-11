# -*- coding: utf-8 -*-
"""
Tách từng ký tự riêng lẻ trong ảnh biển số đã cắt + nhị phân hoá.
Hỗ trợ cả biển 1 dòng (ô tô) và 2 dòng (xe máy).
"""
import cv2
import numpy as np


def segment_characters(binary_image, min_height_ratio=0.35, max_height_ratio=0.6,
                        min_width=4, max_width_ratio=0.6,
                        min_aspect=0.2, max_aspect=1.3):
    """
    Input : ảnh nhị phân (1 kênh) của biển số đã cắt, nét chữ ĐEN trên nền TRẮNG.
    Output: danh sách [{x, y, w, h}, ...] theo thứ tự đọc tự nhiên.

    2 LỚP LỌC quan trọng để loại bỏ VIỀN ĐEN của khung biển số thật (viền
    tạo thành dải trắng mỏng, cao gần hết ảnh sau khi nhị phân hoá + đảo màu):
      - max_height_ratio=0.6   : ký tự thật chỉ cao ~40-44% ảnh; viền cao
                                  tới 75-90% ảnh -> loại được ngay ở đây.
      - min_aspect=0.2         : ký tự thật có tỉ lệ rộng/cao ~0.2-0.5; viền
                                  rất hẹp (0.03-0.15) -> lớp lọc dự phòng thứ 2.
    """
    h_img, w_img = binary_image.shape[:2]
    if h_img == 0 or w_img == 0:
        return []

    inverted = cv2.bitwise_not(binary_image)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        height_ratio = h / float(h_img)
        width_ratio = w / float(w_img)
        aspect = w / float(h) if h else 0.0
        if (min_height_ratio <= height_ratio <= max_height_ratio
                and w >= min_width
                and width_ratio <= max_width_ratio
                and min_aspect <= aspect <= max_aspect):
            boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    if not boxes:
        return []

    boxes.sort(key=lambda b: b["y"])
    avg_h = sum(b["h"] for b in boxes) / len(boxes)

    lines = []
    for b in boxes:
        cy = b["y"] + b["h"] / 2
        placed = False
        for line in lines:
            line_cy = sum(bb["y"] + bb["h"] / 2 for bb in line) / len(line)
            if abs(cy - line_cy) < avg_h * 0.6:
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])

    lines.sort(key=lambda line: sum(bb["y"] for bb in line) / len(line))
    ordered = []
    for line in lines:
        line.sort(key=lambda b: b["x"])
        ordered.extend(line)

    return ordered


def characters_in_lines(image_bgr, line_boxes, pad_ratio=0.12):
    """
    Tách từng KÝ TỰ nằm bên trong các hộp dòng chữ mà PaddleOCR đã tìm ra.
    Hộp dòng đã bó sát chữ (không dính viền khung biển) nên tách sạch hơn nhiều so với
    tách trên cả biển số.

    Input : image_bgr  - ảnh gốc; line_boxes - [{x,y,w,h}] theo toạ độ ảnh gốc.
    Output: [{x,y,w,h}] từng ký tự, toạ độ ảnh gốc. Dòng nào tách không ra thì trả về
            chính hộp dòng đó, để giao diện luôn có vùng để khoanh.
    """
    H, W = image_bgr.shape[:2]
    out = []
    for lb in line_boxes:
        lx, ly, lw, lh = int(lb["x"]), int(lb["y"]), int(lb["w"]), int(lb["h"])
        if lw <= 0 or lh <= 0:
            continue
        pad = max(2, int(lh * pad_ratio))
        x0, y0 = max(0, lx - pad), max(0, ly - pad)
        x1, y1 = min(W, lx + lw + pad), min(H, ly + lh + pad)
        region = image_bgr[y0:y1, x0:x1]
        if region.size == 0:
            continue

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.count_nonzero(th) > th.size / 2:
            th = cv2.bitwise_not(th)  # chữ luôn là phần TRẮNG (ít diện tích hơn nền)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if 0.5 * lh <= h <= 1.2 * lh and w >= 2 and w <= 1.15 * h:
                found.append({"x": x0 + x, "y": y0 + y, "w": w, "h": h})

        if len(found) >= 2:
            found.sort(key=lambda b: b["x"])
            out.extend(found)
        else:
            out.append({"x": lx, "y": ly, "w": lw, "h": lh})
    return out