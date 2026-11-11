import cv2
import numpy as np

img = np.full((480, 640, 3), 200, dtype=np.uint8)
cv2.rectangle(img, (220, 200), (420, 260), (255, 255, 255), -1)
cv2.rectangle(img, (220, 200), (420, 260), (0, 0, 0), 2)
cv2.putText(img, '30A-123.45', (230, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

from ai_module import pipeline

print("Dang xu ly, vui long doi (lan dau co the mat 5-15 giay do import thu vien)...")
result = pipeline.process_frame(img)
print("KET QUA:")
print(result)