import io
import os
import re
import uuid
import json
import datetime
import pytesseract
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Firebase Setup ---
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    cred = credentials.Certificate("capstone-ad4dc-firebase-adminsdk-fbsvc-3005a411f0.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://capstone-ad4dc-default-rtdb.firebaseio.com/'
    })
# cred = credentials.Certificate("capstone-ad4dc-firebase-adminsdk-fbsvc-3005a411f0.json")

# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://capstone-ad4dc-default-rtdb.firebaseio.com/'
# })
db = firestore.client()

# --- FastAPI Setup ---
app = FastAPI(
    title="OCR Receipt Parser + Firebase",
    description="Extract text from receipts and store to Firebase Firestore",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for your Flutter frontend later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USER_ID = "user_satu"  # ideally from authentication token


@app.get("/")
def read_root():
    return {"message": "Welcome to OCR API + Firebase!"}


@app.post("/extract-text/", response_class=JSONResponse)
async def extract_text_and_save(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah bukan gambar.")

    try:
        contents = await image.read()
        np_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="File bukan gambar yang valid")

        # Run OCR
        img = deskew_image(img)
        pil_img = Image.open(io.BytesIO(contents))
        # --- OCR Preprocessing ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        gray = cv2.medianBlur(gray, 3)
        # optional dilation/erosion for thin fonts
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        gray = cv2.erode(gray, kernel, iterations=1)

        # Convert back to PIL for pytesseract
        processed_pil = Image.fromarray(gray)
        cv2.imwrite("processed_image.png", gray)  # for debugging
        text = run_best_ocr(processed_pil)


        if not text.strip():
            raise HTTPException(status_code=400, detail="Tidak ada teks yang terdeteksi pada gambar.")

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        lines = [re.sub(r'[^0-9A-Za-z.,:/\-() ]+', '', l) for l in lines]
        lines = [re.sub(r'\s{2,}', ' ', l).strip() for l in lines]
        # Extract structured data
        merchant = extract_merchant(lines)
        date_str = extract_date(lines)
        total_str = extract_total(lines)
        tunai_str = extract_cash(lines)
        kembali_str = extract_change(lines)
        items = extract_items(lines)

        result = {
            "filename": image.filename,
            "merchant": merchant,
            "date": date_str,
            "total": total_str,
            "tunai": tunai_str,
            "kembali": kembali_str,
            "items": items,
            "raw_lines": lines,
            "raw_text": text,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        # --- Save to Firebase Firestore ---
        doc_ref = db.collection("users").document(USER_ID).collection("expenses").document()
        doc_ref.set(result)

        return JSONResponse(content={"message": "Struk berhasil diproses dan disimpan ke Firebase", "data": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi error: {str(e)}")


# ============================================================
# 💡 Helper functions for receipt parsing
# ============================================================

def extract_merchant(lines):
    for l in lines:
        if any(k in l.lower() for k in ["toko", "mart", "minimarket", "pt", "indomaret", "alfamart", "market", "co", "bengkel", "warung"]):
            return l
    # fallback: first non-numeric line
    for l in lines:
        if not re.search(r'\d', l):
            return l
    return None


def extract_date(lines):
    """Find any date-like pattern."""
    date_pattern = re.compile(r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})')
    for l in lines:
        match = date_pattern.search(l)
        if match:
            return match.group(1)
    return None


def extract_total(lines):
    for l in reversed(lines):
        if re.search(r'total( belanja| harga|)$', l.lower()):
            match = re.search(r'(\d{1,3}(?:[.,]\d{3})+)', l)
            if match:
                return match.group(1)
    return None

def extract_cash(lines):
    """Detect 'tunai' or 'cash'."""
    for l in reversed(lines):
        if "tunai" in l.lower() or "cash" in l.lower():
            match = re.search(r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{0,2})?)', l)
            if match:
                return match.group(1)
    return None


def extract_change(lines):
    """Detect 'kembali' or 'change'."""
    for l in reversed(lines):
        if "kembali" in l.lower() or "change" in l.lower():
            match = re.search(r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{0,2})?)', l)
            if match:
                return match.group(1)
    return None


def extract_items(lines):
    """Extract possible item lines."""
    items = []
    pattern = re.compile(r'(.+?)\s+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{0,2})?)$')
    for line in lines:
        match = pattern.match(line)
        if match:
            name = match.group(1).strip()
            price = match.group(2).replace('.', '').replace(',', '')
            try:
                price = int(price)
                items.append({"item": name, "harga": price})
            except:
                continue
    return items

def deskew_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 255))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def run_best_ocr(img):
    best_text = ""
    for mode in [4, 6, 11]:
        text = pytesseract.image_to_string(img, lang="ind+eng", config=f"--psm {mode}")
        if len(text) > len(best_text):
            best_text = text
    return best_text

