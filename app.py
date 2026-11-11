# -*- coding: utf-8 -*-
"""
Giai đoạn 3 & 4: Giao diện web (Flask) + ghép nối logic.
Chạy: python app.py   -> mở http://localhost:5000
"""
import base64
import io
import threading

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

from database import db
from ai_module import pipeline, detector, ocr_engine

app = Flask(__name__)

# PaddleOCR/YOLO KHÔNG an toàn khi nhiều request chạy song song (Flask dev server mặc định đa
# luồng). Khoá lại để mỗi lúc chỉ xử lý 1 ảnh; request đến sau sẽ chờ.
_scan_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Trang web
# ---------------------------------------------------------------------------
@app.route("/")
def trang_chu():
    return render_template("index.html")


@app.route("/history")
def trang_lich_su():
    return render_template("history.html")


# ---------------------------------------------------------------------------
# Tiện ích chuyển đổi ảnh
# ---------------------------------------------------------------------------
def _decode_upload(file_storage):
    data = file_storage.read()
    npimg = np.frombuffer(data, np.uint8)
    return cv2.imdecode(npimg, cv2.IMREAD_COLOR)


def _decode_base64_dataurl(data_url):
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# API: quét ảnh (upload file HOẶC chụp từ camera dạng base64 dataURL)
# ---------------------------------------------------------------------------
@app.route("/api/scan", methods=["POST"])
def api_scan():
    try:
        if "image_file" in request.files and request.files["image_file"].filename:
            image_bgr = _decode_upload(request.files["image_file"])
        else:
            payload = request.get_json(silent=True) or {}
            data_url = payload.get("image_base64")
            if not data_url:
                return jsonify({"loi": "Không nhận được ảnh (chưa upload file hoặc chưa chụp camera)."}), 400
            image_bgr = _decode_base64_dataurl(data_url)

        if image_bgr is None:
            return jsonify({"loi": "Không đọc được dữ liệu ảnh, vui lòng thử lại."}), 400

        with _scan_lock:
            result = pipeline.process_frame(image_bgr)
        return jsonify(result)

    except Exception as e:
        return jsonify({"loi": f"Lỗi xử lý: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# API: lịch sử nhận diện (danh sách có lọc + chi tiết theo ID)
# ---------------------------------------------------------------------------
@app.route("/api/history", methods=["GET"])
def api_history():
    tu_ngay = request.args.get("tu_ngay") or None
    den_ngay = request.args.get("den_ngay") or None
    bien_so = request.args.get("bien_so") or None
    limit_raw = request.args.get("limit")
    if limit_raw == "all":
        limit = None
    elif limit_raw and limit_raw.isdigit():
        limit = int(limit_raw)
    else:
        limit = 50  # mac dinh: chi tai 50 ban ghi moi nhat, tranh trang Lich su tai qua nhieu anh cung luc
    rows = db.select_lich_su(tu_ngay=tu_ngay, den_ngay=den_ngay, bien_so_tim=bien_so, limit=limit)
    return jsonify(rows)


@app.route("/api/history/<int:record_id>", methods=["GET"])
def api_history_detail(record_id):
    row = db.select_chi_tiet(record_id)
    if row is None:
        return jsonify({"loi": "Không tìm thấy bản ghi."}), 404
    return jsonify(row)


def warm_up_models():
    """
    Nạp sẵn YOLO + PaddleOCR ngay khi khởi động server, thay vì đợi lần quét đầu tiên.
    Trước đây model chỉ được nạp khi có yêu cầu quét đầu tiên nên lần quét đó chậm bất thường
    (nạp model + khởi tạo suy luận lần đầu). Chạy thử trên 1 ảnh tổng hợp CÓ CHỮ để cả mô hình
    phát hiện vùng chữ lẫn mô hình nhận dạng đều chạy qua ít nhất 1 lần (ảnh trắng trơn sẽ không
    kích hoạt mô hình nhận dạng).
    """
    import time
    import cv2
    import numpy as np

    t0 = time.time()
    print("[app] Dang nap san model (YOLO + PaddleOCR), vui long doi, dung mo trinh duyet vua...")
    try:
        img = np.full((300, 480, 3), 120, dtype=np.uint8)
        cv2.rectangle(img, (140, 110), (340, 190), (255, 255, 255), -1)
        cv2.putText(img, "59A-123.45", (150, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        detector.detect(img)                       # nap YOLO + chay thu 1 lan
        crop = img[100:200, 130:350]
        ocr_engine.read_plate_paddleocr(crop)      # nap PaddleOCR + chay thu det + rec
        print(f"[app] San sang sau {time.time() - t0:.0f}s. Mo trinh duyet tai http://localhost:5000")
    except Exception as e:
        # Làm nóng thất bại không được làm sập server: lần quét đầu sẽ chậm như trước.
        print(f"[app] Canh bao: khong lam nong duoc model ({e}); lan quet dau co the cham.")


if __name__ == "__main__":
    db.init_db()
    # use_reloader=False: reloader chạy khối này 2 lần (tiến trình cha + con) -> nạp model 2 lần,
    # và có thể khởi động lại giữa lúc đang xử lý. Sửa code xong tự chạy lại `python app.py`.
    warm_up_models()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)