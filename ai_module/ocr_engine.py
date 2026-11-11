# -*- coding: utf-8 -*-
"""
Giai đoạn 2 - Nhận diện chữ (OCR).
3 phương án theo thứ tự ưu tiên: so khớp mẫu ký tự > Tesseract từng ký tự
> Tesseract đọc cả chuỗi 1 lần (dự phòng cuối, dùng khi tách ký tự thất bại).
Biển 2 dòng (xe máy) sẽ tự thêm dấu cách giữa 2 dòng khi hiển thị.
"""
import re
import cv2
import pytesseract
from pytesseract import Output
from ai_module import char_templates

from paddleocr import PaddleOCR

import os
_DEBUG_PADDLE = os.environ.get("PLATE_OCR_DEBUG") == "1"

_paddle_reader = None


def _get_paddle_reader():
    """
    Khởi tạo PaddleOCR 1 lần duy nhất (lần đầu hơi lâu vì tải/nạp model).
    Từ PaddleOCR 3.x, tham số `use_angle_cls` đã đổi tên thành
    `use_textline_orientation`. Thử tham số mới trước, nếu máy đang cài
    bản cũ (2.x) chưa hỗ trợ tham số mới thì tự rơi về tham số cũ.
    """
    global _paddle_reader
    if _paddle_reader is None:
        # enable_mkldnn=False: né lỗi "NotImplementedError:
        # ConvertPirAttribute2RuntimeAttribute..." — bug đã biết của
        # PaddlePaddle 3.3.x khi chạy suy luận CPU qua oneDNN/MKL-DNN
        # (xem PaddlePaddle/Paddle issue #77340). Chạy CPU không dùng
        # MKL-DNN sẽ chậm hơn 1 chút nhưng ổn định, không crash.
        #
        # use_doc_orientation_classify=False, use_doc_unwarping=False:
        # đây là 2 model dành cho ảnh CHỤP TRANG TÀI LIỆU/GIẤY TỜ (tự
        # xoay, tự nắn cong trang bị vặn). Input của mình là ảnh biển số
        # đã cắt sẵn (không phải trang giấy), để 2 model này bật mặc định
        # sẽ "nắn" nhầm ảnh nhỏ, làm méo/cắt mất chữ ở 2 đầu biển số.
        common_kwargs = dict(
            lang="en",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        try:
            _paddle_reader = PaddleOCR(use_textline_orientation=True, **common_kwargs)
        except TypeError:
            try:
                _paddle_reader = PaddleOCR(use_angle_cls=True, **common_kwargs)
            except TypeError:
                try:
                    _paddle_reader = PaddleOCR(use_angle_cls=True, enable_mkldnn=False, lang="en")
                except TypeError:
                    # Bản PaddleOCR/PaddlePaddle rất cũ, không có các tham số trên
                    _paddle_reader = PaddleOCR(use_angle_cls=True, lang="en")
    return _paddle_reader

_TESS_CONFIG_1_LINE = "--psm 7 --oem 3"
_TESS_CONFIG_MULTI_LINE = "--psm 6 --oem 3"


def _clean_text(raw_text):
    text = raw_text.upper().strip()
    text = re.sub(r"[^0-9A-Z.\- ]", "", text)   # cho phép thêm dấu cách
    text = re.sub(r"-{2,}", "-", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _insert_line_breaks(char_boxes, chars):
    """
    Chèn dấu cách vào giữa các ký tự khi phát hiện chuyển dòng (biển 2 dòng),
    dựa trên khoảng cách tâm-y giữa ký tự hiện tại và ký tự trước đó.
    """
    if not char_boxes:
        return "".join(chars)

    avg_h = sum(b["h"] for b in char_boxes) / len(char_boxes)
    result = [chars[0]]
    prev_cy = char_boxes[0]["y"] + char_boxes[0]["h"] / 2

    for box, ch in zip(char_boxes[1:], chars[1:]):
        cy = box["y"] + box["h"] / 2
        if cy - prev_cy > avg_h * 0.6:  # nhảy xuống dòng dưới
            result.append(" ")
        result.append(ch)
        prev_cy = cy

    return "".join(result)


def read_plate_text(binary_image, is_square=False):
    """Đọc cả chuỗi 1 lần bằng Tesseract (phương án dự phòng cuối)."""
    config = _TESS_CONFIG_MULTI_LINE if is_square else _TESS_CONFIG_1_LINE
    data = pytesseract.image_to_data(binary_image, config=config, output_type=Output.DICT)

    words, confs = [], []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if word and conf >= 0:
            words.append(word)
            confs.append(conf)

    raw_text = " ".join(words) if words else ""
    plate_text = _clean_text(raw_text)
    confidence = round(sum(confs) / len(confs), 2) if confs else 0.0
    return plate_text, confidence


def read_plate_by_characters(binary_image, char_boxes):
    """Đọc từng ký tự riêng bằng Tesseract (--psm 10 = 1 ký tự)."""
    if not char_boxes:
        return "", 0.0

    chars, confs = [], []
    for box in char_boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        pad = max(2, int(h * 0.12))
        y0 = max(0, y - pad)
        y1 = min(binary_image.shape[0], y + h + pad)
        x0 = max(0, x - pad)
        x1 = min(binary_image.shape[1], x + w + pad)
        char_img = binary_image[y0:y1, x0:x1]
        if char_img.size == 0:
            continue

        data = pytesseract.image_to_data(char_img, config="--psm 10 --oem 3", output_type=Output.DICT)
        text_found, conf_found = "", 0.0
        for i, word in enumerate(data["text"]):
            word = word.strip()
            conf = float(data["conf"][i])
            if word and conf >= 0:
                text_found = word[0]
                conf_found = conf
                break

        chars.append(text_found if text_found else "?")
        confs.append(conf_found)

    text = _clean_text(_insert_line_breaks(char_boxes, chars))
    avg_conf = round(sum(confs) / len(confs), 2) if confs else 0.0
    return text, avg_conf


def read_plate_by_template_matching(binary_image, char_boxes, min_avg_score=0.35):
    """Đọc từng ký tự bằng so khớp mẫu (char_templates) — ưu tiên số 1."""
    if not char_boxes:
        return "", 0.0

    chars, scores = [], []
    for box in char_boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        crop = binary_image[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        inverted_crop = cv2.bitwise_not(crop)
        ch, score = char_templates.match_character(inverted_crop)
        chars.append(ch)
        scores.append(score)

    text = _clean_text(_insert_line_breaks(char_boxes, chars))
    avg_score = sum(scores) / len(scores) if scores else 0.0

    if avg_score < min_avg_score:
        return "", 0.0
    return text, round(avg_score * 100, 2)


# ---------------------------------------------------------------------------
# GỢI Ý NÂNG CẤP (tuỳ chọn, không bắt buộc để chạy demo):
#   EasyOCR / PaddleOCR có thể thay thế read_plate_text() nếu cần.
# ---------------------------------------------------------------------------

def _parse_paddle_result_v3(res0):
    """
    PaddleOCR 3.x: mỗi phần tử của result là 1 object giống dict, đọc bằng
    res0["rec_texts"] (list chuỗi) và res0["rec_scores"] (list điểm 0-1).
    Nếu biển số bị tách thành nhiều box chữ (ví dụ "30A" và "-123.45" tách
    riêng), sắp lại theo toạ độ (trên->dưới rồi trái->phải) dựa vào
    rec_polys/dt_polys để ghép lại đúng thứ tự đọc thật.
    """
    try:
        texts = list(res0["rec_texts"])
        scores = list(res0["rec_scores"])
    except (KeyError, TypeError, IndexError):
        return [], []

    polys = None
    for key in ("rec_polys", "dt_polys"):
        try:
            candidate = list(res0[key])
            if len(candidate) == len(texts):
                polys = candidate
                break
        except (KeyError, TypeError):
            continue

    if polys:
        def _sort_key(i):
            poly = polys[i]
            ys = [p[1] for p in poly]
            xs = [p[0] for p in poly]
            return (sum(ys) / len(ys), sum(xs) / len(xs))

        order = sorted(range(len(texts)), key=_sort_key)
        texts = [texts[i] for i in order]
        scores = [scores[i] for i in order]

    return texts, scores


def _parse_paddle_result_v2(res0):
    """
    PaddleOCR 2.x (bản cũ): res0 là list các dòng, mỗi dòng dạng
    [bbox_4_diem, (text, conf)]. Sắp lại theo thứ tự trên->dưới dựa vào
    toạ độ y trung bình của bbox.
    """
    if not isinstance(res0, list):
        return [], []
    try:
        lines_sorted = sorted(res0, key=lambda ln: sum(p[1] for p in ln[0]) / 4)
    except (TypeError, IndexError, KeyError):
        return [], []

    texts, scores = [], []
    for line in lines_sorted:
        try:
            texts.append(line[1][0])
            scores.append(line[1][1])
        except (IndexError, TypeError):
            continue
    return texts, scores


# ---------------------------------------------------------------------------
# Hậu xử lý theo quy tắc biển số VN (chỉ sửa những chỗ chắc chắn):
#   - 2 ký tự đầu (mã tỉnh)      : phải là SỐ
#   - ký tự thứ 3 (seri)         : phải là CHỮ  (VN không dùng I, J, O, Q, R, W)
#   - ký tự thứ 4                : để nguyên (biển xe máy: số; biển đặc biệt như LD, NG: chữ)
#   - từ ký tự thứ 5 trở đi      : phải là SỐ
# Giữ nguyên dấu '-', '.', khoảng trắng của chuỗi gốc.
# ---------------------------------------------------------------------------
_LETTER_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                    "Z": "2", "S": "5", "B": "8", "G": "6"}
_DIGIT_TO_LETTER = {"8": "B", "0": "D", "6": "G", "5": "S", "2": "Z", "4": "A", "1": "L", "7": "T"}


def fix_vietnam_plate_rules(text):
    alnum_count = sum(ch.isalnum() for ch in text)
    if alnum_count < 7 or alnum_count > 9:
        return text  # độ dài bất thường -> không dám sửa

    out, k = [], 0
    for ch in text:
        if not ch.isalnum():
            out.append(ch)
            continue
        if k < 2 or k >= 4:
            ch = _LETTER_TO_DIGIT.get(ch, ch)
        elif k == 2:
            ch = _DIGIT_TO_LETTER.get(ch, ch)
        out.append(ch)
        k += 1
    fixed = "".join(out)
    # Chỉ nhận bản đã sửa khi nó THÀNH biển VN hợp lệ. Nếu vẫn không hợp lệ (biển nước ngoài,
    # bản vẽ, biển đặc biệt...) thì giữ nguyên chữ đọc được, không sửa bừa.
    return fixed if is_valid_vn_plate(fixed) else text


# ---------------------------------------------------------------------------
# Dạng biển số VN hợp lệ: 2 số + 1 chữ + (chữ hoặc số) + 4-5 số,
# vd 59L206377, 30A12345, 51FB4779.
# ---------------------------------------------------------------------------
_VN_PLATE_RE = re.compile(r"^\d{2}[A-Z][A-Z\d]\d{4,5}$")


def is_valid_vn_plate(text):
    return bool(_VN_PLATE_RE.match(re.sub(r"[^0-9A-Z]", "", (text or "").upper())))


def _alnum_len(text):
    return sum(ch.isalnum() for ch in text)


# ---------------------------------------------------------------------------
# Trích từng DÒNG chữ (kèm toạ độ) từ kết quả PaddleOCR, rồi chọn ra những dòng
# thật sự thuộc biển số. Ảnh chụp thật hay dính thêm chữ khác (tên hãng xe
# HONDA, chú thích, số đo...) — không được ghép chúng vào biển số.
# ---------------------------------------------------------------------------
def _box_from_poly(poly):
    try:
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
    except (TypeError, IndexError, ValueError):
        return None
    return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}


def _sort_reading_order(lines):
    """Xếp theo thứ tự đọc: từ trên xuống, cùng hàng thì từ trái sang phải."""
    if not lines or any(l["box"] is None for l in lines):
        return lines  # thiếu toạ độ -> giữ nguyên thứ tự PaddleOCR trả về
    avg_h = sum(l["box"]["h"] for l in lines) / len(lines)

    def cy(l):
        return l["box"]["y"] + l["box"]["h"] / 2

    ordered = sorted(lines, key=cy)
    rows, cur = [], [ordered[0]]
    for l in ordered[1:]:
        if abs(cy(l) - sum(cy(b) for b in cur) / len(cur)) < avg_h * 0.5:
            cur.append(l)
        else:
            rows.append(cur)
            cur = [l]
    rows.append(cur)
    out = []
    for r in rows:
        out.extend(sorted(r, key=lambda l: l["box"]["x"]))
    return out


def _extract_lines(res0):
    """[{"text", "score", "box": {x,y,w,h} hoặc None}] theo thứ tự đọc. Hiểu cả bản 3.x lẫn 2.x."""
    lines = []
    try:  # bản 3.x
        texts = list(res0["rec_texts"])
        scores = list(res0["rec_scores"])
        polys = None
        for key in ("rec_polys", "dt_polys"):
            try:
                cand = list(res0[key])
                if len(cand) == len(texts):
                    polys = cand
                    break
            except (KeyError, TypeError):
                continue
        for i, t in enumerate(texts):
            lines.append({
                "text": t,
                "score": float(scores[i]) if i < len(scores) else 0.0,
                "box": _box_from_poly(polys[i]) if polys else None,
            })
    except (KeyError, TypeError, IndexError):
        lines = []

    if not lines and isinstance(res0, list):  # bản 2.x: [[bbox, (text, conf)], ...]
        for ln in res0:
            try:
                lines.append({"text": ln[1][0], "score": float(ln[1][1]), "box": _box_from_poly(ln[0])})
            except (IndexError, TypeError):
                continue

    lines = [l for l in lines if l["text"] and str(l["text"]).strip()]
    return _sort_reading_order(lines)


def _join_lines(lines, apply_rules):
    text = _clean_text(" ".join(l["text"] for l in lines))
    return fix_vietnam_plate_rules(text) if apply_rules else text


def _starts_like_plate(alnum_text):
    """Biển số VN luôn bắt đầu bằng 2 CHỮ SỐ (mã tỉnh) — lọc bỏ chữ ngẫu nhiên tình cờ
    đúng độ dài (vd 'LEMON.1' dài 6 ký tự nhưng không hề giống biển số)."""
    return len(alnum_text) >= 2 and alnum_text[0].isdigit() and alnum_text[1].isdigit()


def _select_plate_lines(lines, apply_rules):
    """
    Chọn các dòng thuộc biển số:
      1) Có tổ hợp 1-3 dòng liền nhau ghép thành biển VN hợp lệ -> lấy tổ hợp dài nhất.
      2) Không có, nhưng ghép tất cả vẫn ra độ dài 6-9 ký tự VÀ bắt đầu bằng 2 chữ số
         -> giữ hết (như cũ, chỉ thêm điều kiện bắt đầu bằng số).
      3) Còn lại (dính nhiều chữ thừa) -> lấy 1-2 dòng có độ dài gần 8 ký tự nhất, có chữ số
         VÀ bắt đầu bằng 2 chữ số.
      4) Không có gì thoả -> trả về RỖNG (không phải biển số), kèm cờ báo "đã thấy chữ
         nhưng chắc chắn không phải biển" để nơi gọi biết không cần thử lại ở ảnh to hơn.
    """
    n = len(lines)
    if n <= 1:
        return lines

    def mean_score(sub):
        return sum(l["score"] for l in sub) / len(sub)

    valid = []
    for i in range(n):
        for j in range(i, min(n, i + 3)):
            sub = lines[i:j + 1]
            t = _join_lines(sub, apply_rules)
            if is_valid_vn_plate(t):
                valid.append(((_alnum_len(t), mean_score(sub)), sub))
    if valid:
        return max(valid, key=lambda v: v[0])[1]

    whole = re.sub(r"[^0-9A-Z]", "", _join_lines(lines, apply_rules).upper())
    if 6 <= len(whole) <= 9 and _starts_like_plate(whole):
        return lines

    # Dấu '-' là đặc trưng khá riêng của cách trình bày biển số (vd "T80-235.88"), hiếm khi
    # xuất hiện ở các dòng số đo/chú thích khác trong cùng ảnh -> ưu tiên CAO HƠN tiêu chí
    # "bắt đầu bằng 2 chữ số" khi cả hai cùng có ứng viên (tránh chọn nhầm '110'+'520' — 2 con
    # số kích thước tình cờ ghép lại đúng độ dài — thay vì dòng 'T80-235.88').
    cands = []
    for i in range(n):
        for j in range(i, min(n, i + 2)):
            sub = lines[i:j + 1]
            t = _join_lines(sub, apply_rules)
            alnum = re.sub(r"[^0-9A-Z]", "", t.upper())
            if "-" in t and 5 <= len(alnum) <= 10:
                cands.append(((2, _alnum_len(t), mean_score(sub)), sub))
            elif 5 <= len(alnum) <= 9 and _starts_like_plate(alnum):
                cands.append(((1, _alnum_len(t), mean_score(sub)), sub))
    if cands:
        return max(cands, key=lambda c: c[0])[1]
    # Không có tổ hợp nào giống biển số thật (kể cả gần đúng): nhiều khả năng đây không phải
    # ảnh biển số (ảnh sản phẩm, áp phích, nhiều dòng chữ không liên quan...). Trả về RỖNG,
    # không ghép bừa các dòng lại — tốt hơn là báo "không đọc được" còn hơn đọc sai/đọc bậy.
    return []


_OCR_MAX_SIDE = 480  # ảnh đưa vào PaddleOCR không cần lớn: biển số vẫn đọc tốt ở crop nhỏ
                    # (đã đo thực tế ~90% đúng với crop chỉ ~84x59px). Ảnh vùng biển do
                    # YOLO khoanh có khi rất lớn (chụp ảnh nét/độ phân giải cao) khiến mỗi
                    # lần gọi PaddleOCR mất 10-20 giây; thu nhỏ trước giúp nhanh hơn nhiều
                    # mà không giảm độ chính xác đáng kể.


def read_plate_paddleocr(crop_bgr, apply_rules=True, return_lines=False):
    """
    Đọc biển số bằng PaddleOCR. Chỉ giữ những dòng thuộc biển số (bỏ chữ thừa).

    Trả về (chuoi_bien_so, do_tin_cay_0_100); nếu return_lines=True thì thêm phần tử thứ 3:
    danh sách hộp các dòng chữ đã chọn [{x,y,w,h}] theo TOẠ ĐỘ CỦA `crop_bgr` TRUYỀN VÀO
    (tự quy đổi lại nếu bên trong có thu nhỏ ảnh để đọc nhanh hơn).
    """
    reader = _get_paddle_reader()

    h, w = crop_bgr.shape[:2]
    scale = _OCR_MAX_SIDE / float(max(h, w)) if max(h, w) > _OCR_MAX_SIDE else 1.0
    ocr_input = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else crop_bgr
    if scale < 1.0:
        oh, ow = ocr_input.shape[:2]
        print(f"[ocr_engine] thu nho {w}x{h} -> {ow}x{oh} truoc khi dua vao PaddleOCR")

    if hasattr(reader, "predict"):
        try:
            result = reader.predict(ocr_input, text_rec_score_thresh=0.0)
        except TypeError:
            result = reader.predict(ocr_input)
    else:
        result = reader.ocr(ocr_input)

    def pack(text, conf, boxes):
        if scale != 1.0:
            boxes = [{"x": b["x"] / scale, "y": b["y"] / scale, "w": b["w"] / scale, "h": b["h"] / scale}
                     for b in boxes]
        return (text, conf, boxes) if return_lines else (text, conf)

    if not result or result[0] is None:
        return pack("", 0.0, [])

    res0 = result[0]

    if _DEBUG_PADDLE:
        try:
            print("[ocr_engine][debug] rec_texts:", res0["rec_texts"])
            print("[ocr_engine][debug] rec_scores:", res0["rec_scores"])
        except (KeyError, TypeError):
            print("[ocr_engine][debug] raw result:", res0)

    lines = _extract_lines(res0)
    if not lines:
        return pack("", 0.0, [])

    chosen = _select_plate_lines(lines, apply_rules)
    if not chosen:
        # conf=-1.0: cờ báo "đã thấy chữ (lines không rỗng) nhưng chắc chắn không phải
        # biển số" — khác với conf=0.0 (không thấy chữ gì cả, có thể do ảnh nhỏ/mờ, còn
        # đáng thử lại ở độ phân giải cao hơn). Nơi gọi (pipeline.py) dựa vào đây để biết
        # có nên tốn thêm thời gian đọc lại ở ảnh to hơn hay không.
        return pack("", -1.0, [])

    text = _join_lines(chosen, apply_rules)
    conf = round(sum(l["score"] for l in chosen) / len(chosen) * 100, 2)
    return pack(text, conf, [l["box"] for l in chosen if l["box"]])


# ---------------------------------------------------------------------------
# Đọc "bền": đọc bình thường trước; chỉ khi kết quả KHÔNG giống biển số VN mới
# thử lại bằng ảnh phóng to (rồi thêm CLAHE). Đã đo trên 59 ảnh thật: phóng to
# KHÔNG tốt hơn khi dùng cho mọi ảnh (làm hỏng nhiều ca đang đúng), nhưng cứu
# được các ca đọc thiếu/sai bất thường. Nên chỉ dùng làm phương án dự phòng.
# ---------------------------------------------------------------------------
def _upscale(img, factor):
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def _clahe_bgr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


_RETRY_MAX_SIDE = 900  # ảnh sau khi phóng to mà lớn hơn cỡ này thì bỏ qua: chậm mà vô ích


def _rotate(img, angle):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def read_plate_robust(crop_bgr, return_lines=False):
    """
    Trả về (chuoi_bien_so, do_tin_cay_0_100[, hop_cac_dong]). Ca bình thường chỉ tốn 1 lần đọc.
    Chỉ thử phóng to với ảnh NHỎ; ảnh đã to thì phóng thêm không giúp gì mà còn làm chậm.
    Hộp trả về luôn theo toạ độ của ảnh đầu vào (đã quy đổi ngược nếu đã phóng to).
    """
    def pack(t, c, b):
        return (t, c, b) if return_lines else (t, c)

    text, conf, boxes = read_plate_paddleocr(crop_bgr, return_lines=True)
    if is_valid_vn_plate(text):
        return pack(text, conf, boxes)

    h, w = crop_bgr.shape[:2]
    candidates = [(text, conf, boxes)]
    for factor, use_clahe in ((2, False), (2, True)):
        if max(h, w) * factor > _RETRY_MAX_SIDE:
            continue
        img = _upscale(crop_bgr, factor)
        if use_clahe:
            img = _clahe_bgr(img)
        t, c, b = read_plate_paddleocr(img, return_lines=True)
        b = [{"x": bb["x"] / factor, "y": bb["y"] / factor, "w": bb["w"] / factor, "h": bb["h"] / factor}
             for bb in b]
        if is_valid_vn_plate(t):
            return pack(t, c, b)
        candidates.append((t, c, b))

    # Vẫn chưa ra biển hợp lệ: thử xoay ảnh. deskew (preprocess.py) cố tình chỉ sửa góc nghiêng
    # NHẸ (2-15 độ) nên không xử lý ảnh xoay hẳn 90 độ hoặc lộn ngược 180 độ.
    # Chỉ thử những hướng HỢP LÝ về hình học để không tốn thêm lần gọi OCR vô ích:
    #  - crop "cao hơn rộng" (aspect < 1): biển số thật luôn nằm ngang, nên đây gần như chắc chắn
    #    là biển bị xoay 90 độ -> thử xoay thuận và ngược chiều kim đồng hồ.
    #  - crop nằm ngang: nếu bị ngược thì chỉ có thể là lộn ngược 180 độ -> thử đúng 1 hướng đó.
    aspect = w / float(h) if h else 1.0
    rotation_order = (90, 270) if aspect < 1.0 else (180,)
    for angle in rotation_order:
        rotated = _rotate(crop_bgr, angle)
        t, c, _b = read_plate_paddleocr(rotated, return_lines=True)
        if is_valid_vn_plate(t):
            # Hộp ký tự tính trên ảnh đã xoay, không map ngược về ảnh gốc (ca hiếm, dự phòng
            # cuối) -> trả rỗng: nơi gọi không vẽ khung ký tự riêng cho ca này nhưng vẫn
            # hiển thị đúng chữ + độ tin cậy.
            return pack(t, c, [])
        candidates.append((t, c, []))

    # Không cách nào ra biển hợp lệ: giữ kết quả dài nhất (đọc được nhiều ký tự nhất)
    return pack(*max(candidates, key=lambda tcb: _alnum_len(tcb[0])))