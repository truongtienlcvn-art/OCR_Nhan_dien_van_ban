# -*- coding: utf-8 -*-
"""
Đánh giá OCR trên bộ ảnh THẬT có nhãn (training/dataset_ocr_test/...).
Chạy đúng luồng của app: YOLO detect -> cắt -> tiền xử lý -> PaddleOCR,
nhưng KHÔNG ghi database, KHÔNG lưu ảnh vào static/.

Cách dùng (chạy ở thư mục gốc dự án, đã bật .venv):
    python eval_ocr.py          -> 20 ảnh đầu
    python eval_ocr.py 60       -> 60 ảnh đầu (bộ này đã dùng để thiết kế thuật toán)
    python eval_ocr.py rest     -> các ảnh SAU ảnh thứ 59: bộ CHƯA từng dùng để chỉnh,
                                   cho con số trung thực nhất
    python eval_ocr.py all      -> toàn bộ

So 2 cách, cả hai đều qua deskew + hậu xử lý quy tắc biển VN như app thật:
    B = đọc 1 lần (cách cũ)
    R = read_plate_robust: đọc 1 lần, chỉ thử lại bằng phóng to khi kết quả
        không giống biển số VN (cách app đang dùng)
Kết quả ghi vào eval_results.csv.
"""
import csv
import glob
import os
import re
import sys
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "training", "dataset_ocr_test", "biensoxemayhon100bien")
IMG_DIR = os.path.join(DATA_DIR, "anh")
LABEL_DIR = os.path.join(DATA_DIR, "label")


def imread_unicode(path):
    """cv2.imread lỗi với đường dẫn có dấu tiếng Việt -> đọc bằng numpy."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def normalize(s):
    return re.sub(r"[^0-9A-Z]", "", (s or "").upper())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(pred, truth):
    if not truth:
        return 0.0
    return max(0.0, 1.0 - edit_distance(pred, truth) / len(truth))


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "20"
    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not files:
        print(f"Khong thay anh trong: {IMG_DIR}")
        return
    if arg == "rest":
        files = files[59:]
    elif arg != "all":
        files = files[:int(arg)]

    print("Dang nap YOLO + PaddleOCR (lan dau mat vai giay)...")
    from ai_module import detector, preprocess, ocr_engine

    names = ["B", "R"]
    rows = []
    ok = {k: 0 for k in names}
    acc = {k: 0.0 for k in names}
    n = 0
    t0 = time.time()

    def read_once(im):
        return normalize(ocr_engine.read_plate_paddleocr(im)[0])

    def read_robust(im):
        return normalize(ocr_engine.read_plate_robust(im)[0])

    for path in files:
        name = os.path.basename(path)
        label_path = os.path.join(LABEL_DIR, name.replace(".jpg", ".txt"))
        if not os.path.exists(label_path):
            continue
        with open(label_path, encoding="utf-8") as f:
            truth = normalize(f.read())

        img = imread_unicode(path)
        if img is None:
            continue

        boxes = detector.detect(img)
        preds = {k: "NO_DETECT" for k in names}
        src = "-"
        if boxes:
            best = sorted(boxes, key=lambda b: (b["score"], b["w"] * b["h"]), reverse=True)[0]
            src = best.get("source", "?")
            x, y = max(0, best["x"]), max(0, best["y"])
            crop = img[y:y + best["h"], x:x + best["w"]]
            if crop.size == 0:
                preds = {k: "EMPTY_CROP" for k in names}
            else:
                display, _ = preprocess.enhance_for_ocr(crop)
                preds = {"B": read_once(display), "R": read_robust(display)}

        n += 1
        for k in names:
            ok[k] += preds[k] == truth
            acc[k] += char_accuracy(preds[k], truth)
        rows.append([name, truth] + [preds[k] for k in names] + [src])
        marks = " ".join(f"{k}[{'OK ' if preds[k] == truth else 'SAI'}]" for k in names)
        detail = "" if all(preds[k] == truth for k in names) else "  " + " | ".join(f"{k}={preds[k]}" for k in names)
        print(f"{n:3d} {name:20s} dap_an={truth:10s} {marks}{detail}")

    if n == 0:
        print("Khong co anh nao co nhan.")
        return

    labels_txt = {"B": "B doc 1 lan (cu)        ", "R": "R doc ben (app dang dung)"}
    print("\n" + "=" * 70)
    print(f"So anh: {n}   Thoi gian: {time.time() - t0:.0f}s")
    for k in names:
        print(f"{labels_txt[k]}: dung hoan toan {ok[k]:3d}/{n} = {ok[k] / n:4.0%} | chinh xac ky tu TB {acc[k] / n:4.0%}")
    print("Luu y: nhan anh 0000_02187_b bi sai (bien thuc te la 59E121500); neu co trong bo chay thi cong them 1 anh dung.")

    out_csv = os.path.join(BASE, "eval_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["anh", "dap_an", "B_doc_1_lan", "R_doc_ben", "nguon_detect"])
        w.writerows(rows)
    print(f"Da luu chi tiet vao {out_csv}")


if __name__ == "__main__":
    main()