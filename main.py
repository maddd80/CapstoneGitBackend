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

cred = credentials.Certificate("capstone-ad4dc-firebase-adminsdk-fbsvc-3005a411f0.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://capstone-ad4dc-default-rtdb.firebaseio.com/'
})
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
        pil_img = Image.open(io.BytesIO(contents))
        text = pytesseract.image_to_string(pil_img, lang="ind+eng")

        if not text.strip():
            raise HTTPException(status_code=400, detail="Tidak ada teks yang terdeteksi pada gambar.")

        lines = [l.strip() for l in text.splitlines() if l.strip()]

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
    """Detect possible merchant name (top or bottom)."""
    candidates = lines[:5] + lines[-5:]
    for l in candidates:
        if any(k in l.lower() for k in ["toko", "mart", "pt", "indomaret", "alfamart", "malang", "market", "co", "minimarket"]):
            return l
    return candidates[0] if candidates else None


def extract_date(lines):
    """Find any date-like pattern."""
    date_pattern = re.compile(r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})')
    for l in lines:
        match = date_pattern.search(l)
        if match:
            return match.group(1)
    return None


def extract_total(lines):
    """Detect 'total belanja' or 'harga jual'."""
    for l in reversed(lines):
        if re.search(r'total|harga jual|jumlah|total belanja', l.lower()):
            match = re.search(r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{0,2})?)', l)
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
