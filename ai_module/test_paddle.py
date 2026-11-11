import cv2
from ai_module import ocr_engine

# Đổi đường dẫn này thành 1 ảnh biển số bất kỳ bạn có sẵn trên máy
img = cv2.imread("duong_dan_anh_bien_so.jpg")
text, conf = ocr_engine.read_plate_paddleocr(img)
print("KET QUA:", text, "| Do tin cay:", conf)