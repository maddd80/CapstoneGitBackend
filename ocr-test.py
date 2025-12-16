import cv2
import pytesseract

img = cv2.imread("contoh_struk.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray, config="--psm 6")
print(text)