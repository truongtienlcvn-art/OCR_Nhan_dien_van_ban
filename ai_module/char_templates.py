# -*- coding: utf-8 -*-
"""
Bộ mẫu ký tự (2 kiểu chữ: đứng + nghiêng) lấy từ ảnh font thật của biển số
VN, dùng để nhận diện bằng so khớp mẫu (template matching).
Cần có file: ai_module/assets/plate_font_templates.png

QUAN TRỌNG: cả mẫu (template) và ký tự cắt được lúc chạy đều phải được
"cắt sát viền" (bỏ hết khoảng trắng thừa quanh nét chữ) trước khi resize về
cùng kích thước chuẩn rồi mới so sánh — nếu không, độ "đệm" xung quanh khác
nhau giữa 2 bên sẽ làm sai lệch điểm số so khớp (đã kiểm chứng thực tế: bỏ
qua bước này khiến '8' bị nhận nhầm thành 'B' with độ tin cậy cao).
"""
import os
import cv2
import numpy as np

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SHEET_PATH = os.path.join(_BASE_DIR, "assets", "plate_font_templates.png")

TEMPLATE_W, TEMPLATE_H = 20, 30
GRID_COLS, GRID_ROWS = 6, 6
CHARSET_ORDER = "ABCDEFGHKLMNPQRSTUVXYZ0123456789-."
N_STYLES = 2

_templates_cache = None


def _tight_crop_resize(binary_img, size=(TEMPLATE_W, TEMPLATE_H)):
    """
    Cắt sát viền vùng có nét (bỏ hết khoảng trắng thừa xung quanh), rồi mới
    resize về kích thước chuẩn. Đảm bảo mẫu và ký tự thật luôn được chuẩn
    hoá theo đúng cùng 1 cách trước khi so sánh.
    """
    coords = cv2.findNonZero(binary_img)
    if coords is None:
        return cv2.resize(binary_img, size, interpolation=cv2.INTER_AREA)
    x, y, w, h = cv2.boundingRect(coords)
    tight = binary_img[y:y + h, x:x + w]
    resized = cv2.resize(tight, size, interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return resized


def _load_templates():
    sheet = cv2.imread(_SHEET_PATH, cv2.IMREAD_GRAYSCALE)
    if sheet is None:
        raise FileNotFoundError(
            f"Không tìm thấy bộ mẫu ký tự tại: {_SHEET_PATH}. "
            f"Hãy đặt file plate_font_templates.png vào ai_module/assets/."
        )
    templates = {ch: [] for ch in CHARSET_ORDER}
    style_h = GRID_ROWS * TEMPLATE_H
    for style_idx in range(N_STYLES):
        y_base = style_idx * style_h
        for i, ch in enumerate(CHARSET_ORDER):
            r, c = divmod(i, GRID_COLS)
            crop = sheet[y_base + r * TEMPLATE_H: y_base + (r + 1) * TEMPLATE_H,
                         c * TEMPLATE_W: (c + 1) * TEMPLATE_W]
            _, binary = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)
            normalized = _tight_crop_resize(binary)  # <-- chuẩn hoá ngay khi nạp mẫu
            templates[ch].append(normalized)
    return templates


def get_templates():
    global _templates_cache
    if _templates_cache is None:
        _templates_cache = _load_templates()
    return _templates_cache


def match_character(char_binary_image):
    """
    Input : ảnh nhị phân 1 ký tự đã cắt — chữ TRẮNG trên nền ĐEN.
    Output: (ký_tự_đoán_được, điểm_tương_đồng 0.0-1.0)
    """
    if char_binary_image.size == 0:
        return "?", 0.0

    resized = _tight_crop_resize(char_binary_image)  # <-- chuẩn hoá giống hệt lúc nạp mẫu

    best_char, best_score = "?", -1.0
    for ch, variants in get_templates().items():
        for tmpl in variants:
            inter = cv2.bitwise_and(resized, tmpl)
            union = cv2.bitwise_or(resized, tmpl)
            inter_count = int(np.count_nonzero(inter))
            union_count = int(np.count_nonzero(union))
            score = inter_count / union_count if union_count > 0 else 0.0
            if score > best_score:
                best_score, best_char = score, ch

    return best_char, best_score