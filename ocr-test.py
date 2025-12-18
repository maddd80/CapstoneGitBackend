import cv2

try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
except Exception as e:
    raise RuntimeError("PaddleOCR not available. Install with: pip install paddleocr paddlepaddle") from e

img = cv2.imread("contoh_struk.jpg")
if img is None:
    raise FileNotFoundError("contoh_struk.jpg not found")

result = ocr.ocr(img, cls=True)
texts = []
for line in result:
    if isinstance(line, list):
        for rec in line:
            try:
                texts.append(rec[1][0])
            except Exception:
                pass
    else:
        try:
            texts.append(str(line))
        except Exception:
            pass

print("\n".join(texts))