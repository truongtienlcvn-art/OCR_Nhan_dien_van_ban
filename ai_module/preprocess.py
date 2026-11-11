# -*- coding: utf-8 -*-
"""
Giai đoạn 2 - Tiền xử lý ảnh (OpenCV), đúng theo yêu cầu đề bài:
  1) Chuyển ảnh biển số đã cắt sang ảnh xám.
  2) Tăng độ tương phản (CLAHE).
  3) Nhị phân hoá (adaptive threshold) để làm rõ nét chữ.
  4) Nếu biển số bị nghiêng: dùng minAreaRect + biến đổi phối cảnh để kéo thẳng.
"""
import cv2
import numpy as np


def _estimate_skew_angle(crop_bgr):
    """
    Ước lượng góc nghiêng (độ) của vùng chữ, chuẩn hoá về khoảng (-45, 45].
    minAreaRect ĐỔI QUY ƯỚC GÓC giữa các phiên bản OpenCV (khi -90..0, khi
    0..90; biển thẳng có thể trả 0, 90 hoặc -90). Nên chuẩn hoá bằng modulo 90
    thay vì so sánh cứng, để không bao giờ xoay nhầm ~90 độ.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None or len(coords) < 20:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    return ((angle + 45.0) % 90.0) - 45.0


def deskew(crop_bgr, max_angle=15.0, min_angle=2.0):
    """
    Kéo thẳng biển số nếu bị nghiêng nhẹ. Chỉ xoay khi góc nằm trong
    [min_angle, max_angle] độ: dưới min_angle coi như thẳng; trên max_angle
    thì khả năng cao là ước lượng sai (nhiễu nền/viền), xoay sẽ làm hỏng ảnh.
    """
    angle = _estimate_skew_angle(crop_bgr)
    if abs(angle) < min_angle or abs(angle) > max_angle:
        return crop_bgr

    (h, w) = crop_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        crop_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def enhance_for_ocr(crop_bgr):
    """
    Trả về (ảnh_da_xu_ly_de_hien_thi_BGR, ảnh_nhi_phan_de_OCR)
    """
    straightened = deskew(crop_bgr)

    # Phóng to nếu ảnh biển số quá nhỏ để Tesseract đọc tốt hơn
    h, w = straightened.shape[:2]
    if h < 60:
        scale = 60.0 / h
        straightened = cv2.resize(
            straightened, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.cvtColor(straightened, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    denoised = cv2.bilateralFilter(contrast, 7, 50, 50)

    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )

    # Loại nhiễu nhỏ bằng phép mở hình thái học
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return straightened, binary