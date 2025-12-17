import io
import os
import re
import uuid
import json
import datetime
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import tempfile 
import calendar

# --- NEW: PADDLE OCR ---
from paddleocr import PaddleOCR

# --- Firebase Setup ---
import firebase_admin
from firebase_admin import credentials, firestore

# --- KONEKSI FIREBASE ---
if not firebase_admin._apps:
    try:
        cred_path = r"capstone-ad4dc-firebase-adminsdk-fbsvc-3005a411f0.json"
        cred = credentials.Certificate(cred_path) 
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://capstone-ad4dc-default-rtdb.firebaseio.com/'
        })
    except Exception as e:
        print(f"Gagal koneksi ke Firebase: {e}")
        
db = firestore.client() 

# --- FastAPI Setup ---
app = FastAPI(title="PaddleOCR Receipt Parser", version="10.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- 🚀 INITIALIZE PADDLE OCR (Global Load) ---
# lang='en' is usually better for receipts (numbers/prices) than 'id'
# use_angle_cls=True allows it to read upside-down or rotated receipts
print("Loading PaddleOCR Model... (This takes a few seconds on startup)")
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False) 

# ==========================================
# 1. HELPER FUNCTIONS (Geometry Only)
# ==========================================
# We KEEP the cropping logic because it's useful, but REMOVE the aggressive cleanup.

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def preprocess_for_paddle(image_contents: bytes):
    """
    Simplified Preprocessing for AI OCR.
    Paddle prefers Color/Grayscale images. It HATES binary thresholding.
    """
    # 1. Load
    np_img = np.frombuffer(image_contents, np.uint8)
    original_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    # 2. Smart Crop (Detect Receipt Paper)
    # (Same logic as before, just keeps the receipt)
    height, width = original_img.shape[:2]
    scale_factor = 1.0
    if width > 600:
        scale_factor = 600 / width
        small_img = cv2.resize(original_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    else:
        small_img = original_img.copy()
    
    gray_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_small, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    receipt_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            receipt_contour = approx
            break

    final_img = original_img
    if receipt_contour is not None:
        receipt_contour = receipt_contour.reshape(4, 2) / scale_factor
        final_img = four_point_transform(original_img, receipt_contour)

    # 3. Resize if too massive (Paddle is slow on 4K images)
    # We resize to width ~1200px which is accurate and fast
    h, w = final_img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        final_img = cv2.resize(final_img, (1200, int(h * scale)))
        
    return final_img


# ==========================================
# 2. MAIN LOGIC (Paddle Version)
# ==========================================

def process_and_save_sync_paddle(image_contents: bytes, filename: str, userId: str, projectId: str, category: str):
    try:
        # A. Preprocess (Crop & Resize only)
        # We DO NOT binarize. Paddle loves color/gray.
        img = preprocess_for_paddle(image_contents)
        
        # B. Run PaddleOCR
        # result structure: [[[[box], [text, confidence]], ...]]
        result = ocr_engine.ocr(img, cls=True)
        
        # C. Extract Text
        raw_lines = []
        if result and result[0]:
            # Sort boxes by Y (top to bottom) is usually handled by Paddle, 
            # but getting the raw string list is enough for your Regex parser.
            for line in result[0]:
                text_detected = line[1][0]
                # Optional: Filter low confidence garbage
                confidence = line[1][1]
                if confidence > 0.6: 
                    raw_lines.append(text_detected)
        
        full_text = "\n".join(raw_lines)
        
        if not full_text.strip():
            full_text = "Tidak ada teks terdeteksi."

        # D. Reuse your existing 'Brain' (Regex Parser)
        # This is the beauty of it: The "dumb" regex now gets "smart" text input.
        parsed_data = parse_receipt_with_regex_v4_0(full_text) 

        # E. Prepare & Save
        result_data = {
            "filename": filename,
            "merchant": parsed_data.get("merchant"),
            "date": parsed_data.get("date"),
            "total": parsed_data.get("total_int"),
            "total_string": parsed_data.get("total_str"),
            "tunai": parsed_data.get("tunai_int"),
            "kembali": parsed_data.get("kembali_int"),
            "items": parsed_data.get("items"),
            "raw_text": full_text, # Good for debugging
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "uploadedBy": userId,
            "category": category
        }
        
        doc_ref = db.collection("projects").document(projectId).collection("expenses").document()
        doc_ref.set(result_data) 
        
        result_data["timestamp"] = result_data["timestamp"].isoformat()
        return result_data

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise e 


@app.post("/extract-text/", response_class=JSONResponse)
async def extract_text_and_save( 
    image: UploadFile = File(...),
    userId: str = Form(...),
    projectId: str = Form(...),
    category: str = Form(...) 
):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Not an image.")

    try:
        contents = await image.read()
        # Call the new Paddle function
        result = await run_in_threadpool(
            process_and_save_sync_paddle, 
            image_contents=contents,
            filename=image.filename,
            userId=userId,
            projectId=projectId,
            category=category
        )
        return JSONResponse(content={"message": "Success (PaddleOCR)", "data": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 3. EXISTING REGEX PARSER (Unchanged)
# ==========================================
# Paste your existing 'parse_receipt_with_regex_v4_0' and 'clean_price' functions here.
# PaddleOCR output format matches what this function expects (lines of text).
# ...
def clean_price(price_str):
    if not price_str: return 0
    try:
        cleaned = re.sub(r'\D', '', price_str)
        return int(cleaned)
    except (ValueError, TypeError): 
        return 0

MONTH_MAP = {name.lower(): num for num, name in enumerate(calendar.month_abbr) if num}
MONTH_MAP['mar'] = 3

def parse_receipt_with_regex_v4_0(text):
    # ... (Semua kode parsing regex-mu yang sudah v4.0) ...
    lines = text.splitlines()
    total_int = 0; total_str = None; merchant = None; date = None; items = []; tunai_int = 0; tunai_str = None; kembali_int = 0; kembali_str = None
    
    # UPDATED TOTAL PATTERN:
    # Matches: TOTAL, TOTAI, TWAL, T0TAL, TAGIHAN, JUMLAH, HARGA JUAL (Indomaret specific)
    total_pattern = re.compile(r'(?i)(To[tTl][aA4]l?|TWAL|JUMLAH|TAGIHAN|HARGA\s*JUAL|GRAND\s*TOTAL)\s*[:\->\s|«;]*\s*([\d.,\s]+)')
    
    # UPDATED TUNAI PATTERN:
    # Matches: TUNAI, TUMAI, TUHAI, CASH
    tunai_pattern = re.compile(r'(?i)(TU[NMH]AI|CASH|BAYAR|DIBAYAR)\s*[:\->\s|«;]*\s*([\d.,\s]+)')
    kembali_pattern = re.compile(r'(?i)(KEMBA[LI]?|CHANGE)\s*[:\->\s|«;]*\s*([\d.,\s]+)')

    date_pattern_1 = re.compile(r'(\d{2}/\d{2}/\d{4})') 
    date_pattern_2 = re.compile(r'(\d{2}\.\d{2}\.\d{2,4})')
    date_pattern_3 = re.compile(r'(\d{2}-\d{2}-\d{2,4})')
    date_pattern_4 = re.compile(r'(?i)Tangga[l]?\s*:\s*(\S+)') 
    date_pattern_5 = re.compile(r'(\d{2}\.\d{2}\.\d{2})') 
    date_pattern_6 = re.compile(r'(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})')

    item_pattern_1 = re.compile(r'(?i)^(\d+)\s*[xX]\s+(.+?)\s+([\d.,]+)$')
    item_pattern_2 = re.compile(r'^([a-zA-Z\s\d/.]+?)\s+(\d+)\s+([\d.,]+)\s+([\d.,]+)$')
    item_pattern_3 = re.compile(r'^([a-zA-Z][a-zA-Z\s./-]{3,})\s+([\d.,]{3,})$')
    item_pattern_4 = re.compile(r'^(\d+)\s+(.+?)\s+(\d+)\s+([\d.,]+)\s+(?:[\d.,]+\s+)?([\d.,]+)$')

    FILTER_KEYWORDS = ["TOTAL", "TUNAI", "KEMBALI", "PPN", "TAX", "SUBTOTAL", "DISKON", "NPWP", "HARGA JUAL", "ITEM", "CASHIER", "ADMINISTRATOR"]
    
    for line in lines:
        line_clean = line.strip()
        if not merchant and len(line_clean) > 3 and any(c.isalpha() for c in line_clean):
            if "NO #" not in line_clean.upper() and "KIN ." not in line_clean.upper():
                    merchant = line_clean
            if "indomaret" in line_clean.lower() or "super indo" in line_clean.lower() or "alfamart" in line_clean.lower() or "lilac" in line_clean.lower():
                merchant = line_clean
                break 

    for line in lines:
        line = line.strip()
        if not line: continue 

        if not date:
            match_6 = date_pattern_6.search(line)
            if match_6:
                try:
                    day = match_6.group(1).zfill(2)
                    month_name = match_6.group(2).lower()
                    month_num = str(MONTH_MAP.get(month_name, "01")).zfill(2)
                    year = match_6.group(3)
                    if month_num != "01":
                        date = f"{day}-{month_num}-{year}"
                        continue
                except Exception as e: 
                    print(f"Error parsing tanggal: {e}")
                    pass
            
            match = date_pattern_1.search(line) or \
                    date_pattern_2.search(line) or \
                    date_pattern_3.search(line) or \
                    date_pattern_4.search(line) or \
                    date_pattern_5.search(line)
            if match and "NO #" not in line.upper():
                date = match.group(1).replace('.', '-').replace('/', '-')
                continue 

        total_match = total_pattern.search(line)
        if total_match: total_str = total_match.group(2); total_int = clean_price(total_str); continue
        tunai_match = tunai_pattern.search(line)
        if tunai_match: tunai_str = tunai_match.group(2); tunai_int = clean_price(tunai_str); continue
        kembali_match = kembali_pattern.search(line)
        if kembali_match: kembali_str = kembali_match.group(2); kembali_int = clean_price(kembali_str); continue

        is_keyword = any(keyword in line.upper() for keyword in FILTER_KEYWORDS)
        if is_keyword:
            continue 

        match_4 = item_pattern_4.search(line)
        if match_4:
            try:
                name = match_4.group(2).strip()
                qty = int(match_4.group(3))
                price = clean_price(match_4.group(5))
                unit_price = clean_price(match_4.group(4))
                if price > 0: items.append({"name": name, "qty": qty, "price": price, "unit_price": unit_price})
                continue 
            except Exception: pass

        match_1 = item_pattern_1.search(line)
        if match_1:
            try:
                qty = int(match_1.group(1))
                name = match_1.group(2).strip()
                price = clean_price(match_1.group(3))
                unit_price = price / qty if qty > 0 else 0
                if price > 0: items.append({"name": name, "qty": qty, "price": price, "unit_price": unit_price})
                continue 
            except Exception: pass

        match_2 = item_pattern_2.search(line)
        if match_2:
            try:
                name = match_2.group(1).strip()
                qty = int(match_2.group(2))
                unit_price = clean_price(match_2.group(3))
                price = clean_price(match_2.group(4)) 
                if price > 0: items.append({"name": name, "qty": qty, "price": price, "unit_price": unit_price})
                continue 
            except Exception: pass
            
        match_3 = item_pattern_3.search(line)
        if match_3:
            try:
                name = match_3.group(1).strip()
                price = clean_price(match_3.group(2))
                qty = 1 
                if price > 0: items.append({"name": name, "qty": qty, "price": price, "unit_price": price})
                continue 
            except Exception: pass
            
    return {"merchant": merchant, "date": date, "total_str": total_str, "total_int": total_int, "items": items, "tunai_str": tunai_str, "tunai_int": tunai_int, "kembali_str": kembali_str, "kembali_int": kembali_int}
