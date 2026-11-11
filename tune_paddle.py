# -*- coding: utf-8 -*-
"""
Script THỬ NGHIỆM (không phải code chính thức): chạy PaddleOCR trên 1 ảnh
biển số với NHIỀU tổ hợp tham số cùng lúc, in bảng kết quả để chọn ra cấu
hình đọc đúng nhất trong 1 lần chạy duy nhất — tránh phải sửa code, chạy,
gửi kết quả, sửa tiếp... nhiều vòng.

Cách dùng:
    python tune_paddle.py
        -> dùng ảnh test giả có sẵn (chữ "30A-123.45" vẽ bằng cv2.putText)

    python tune_paddle.py duong/dan/anh_bien_so_that.jpg
        -> dùng 1 ảnh biển số THẬT của bạn (khuyên dùng cái này hơn, vì ảnh
           giả có thể có đặc điểm hình học khác ảnh thật, làm sai lệch kết
           luận — chọn 1 ảnh trong static/images/originals/ mà bạn biết
           chắc biển số thật là gì để so sánh).

Sau khi chạy xong, gửi lại TOÀN BỘ bảng kết quả in ra (không cần gửi lại
từng dòng riêng lẻ) để chọn đúng cấu hình rồi cố định vào ocr_engine.py.
"""
import itertools
import sys

import cv2
import numpy as np


def _make_fake_test_image():
    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.rectangle(img, (220, 200), (420, 260), (255, 255, 255), -1)
    cv2.rectangle(img, (220, 200), (420, 260), (0, 0, 0), 2)
    cv2.putText(img, "30A-123.45", (230, 240), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def _get_crop(img):
    """
    Lấy đúng crop biển số bằng CHÍNH pipeline detect + tiền xử lý của dự án
    (ai_module.detector + ai_module.preprocess), để kết quả thử nghiệm khớp
    với thực tế khi chạy qua app.py, không thử trên ảnh khác đi.
    """
    from ai_module import detector, preprocess
    boxes = detector.detect(img)
    if not boxes:
        raise RuntimeError("Không detect được biển số nào trong ảnh này.")
    best = sorted(boxes, key=lambda b: (b["score"], b["w"] * b["h"]), reverse=True)[0]
    x, y, w, h = best["x"], best["y"], best["w"], best["h"]
    x, y = max(0, x), max(0, y)
    crop = img[y:y + h, x:x + w]
    straightened, _ = preprocess.enhance_for_ocr(crop)
    return straightened


def _pad(crop, ratio):
    if ratio <= 0:
        return crop
    h, w = crop.shape[:2]
    py, px = int(h * ratio), int(w * ratio)
    return cv2.copyMakeBorder(crop, py, py, px, px, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        img = cv2.imread(path)
        if img is None:
            print(f"Khong doc duoc anh: {path}")
            return
        print(f"Dung anh that: {path}")
    else:
        img = _make_fake_test_image()
        print("Khong truyen duong dan anh -> dung anh test gia (30A-123.45)")

    crop = _get_crop(img)
    print(f"Kich thuoc crop: {crop.shape[1]}x{crop.shape[0]}")
    cv2.imwrite("tune_crop_debug.jpg", crop)
    print("Da luu crop vao tune_crop_debug.jpg (mo xem lai neu can).\n")

    from paddleocr import PaddleOCR
    print("Dang khoi tao PaddleOCR (mat vai giay, chi 1 lan)...")
    try:
        reader = PaddleOCR(
            use_textline_orientation=True, lang="en", enable_mkldnn=False,
            use_doc_orientation_classify=False, use_doc_unwarping=False,
        )
    except TypeError:
        reader = PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=False)
    print("Da san sang. Bat dau thu cac to hop tham so...\n")

    pad_ratios = [0.0, 0.15]
    box_threshs = [0.3, 0.6]
    unclip_ratios = [1.5, 2.5, 4.0]

    header = f"{'pad':>5} | {'box_thresh':>10} | {'unclip':>7} | ket qua doc duoc"
    print(header)
    print("-" * len(header) * 2)

    for pad_ratio, box_thresh, unclip in itertools.product(pad_ratios, box_threshs, unclip_ratios):
        test_img = _pad(crop, pad_ratio)
        try:
            result = reader.predict(
                test_img,
                text_det_box_thresh=box_thresh,
                text_det_unclip_ratio=unclip,
                text_rec_score_thresh=0.0,
            )
        except TypeError as e:
            print(f"{pad_ratio:>5} | {box_thresh:>10} | {unclip:>7} | (loi tham so: {e})")
            continue

        if not result or result[0] is None:
            texts = []
        else:
            res0 = result[0]
            try:
                texts = list(res0["rec_texts"])
            except (KeyError, TypeError):
                texts = []

        print(f"{pad_ratio:>5} | {box_thresh:>10} | {unclip:>7} | {texts}")

    print("\nXong. Gui lai toan bo bang ket qua o tren.")


if __name__ == "__main__":
    main()