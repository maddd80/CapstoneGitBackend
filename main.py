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
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import subprocess 
import tempfile 
import calendar

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
        print(f"Gagal koneksi ke Firebase. Pastikan file serviceAccountKey.json ada di path: {cred_path}")
        print(f"Error: {e}")
        
db = firestore.client() 

# --- FastAPI Setup ---
app = FastAPI(
    title="OCR Receipt Parser - Project Based",
    version="9.0.0 (Manual Category)" # Versi baru
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# --- 🔥 HAPUS FUNGSI KATEGORI (KARENA DIHANDLE FLUTTER) ---
# CATEGORIES = { ... }
# def get_category_from_text(text):
#     ...
# --- 🔥 AKHIR BLOK HAPUS ---


@app.get("/")
def read_root():
    return {"message": "Welcome to Project-Based OCR API (Manual Category)!"}


import numpy as np
import cv2
from PIL import Image

# ==========================================
# 🔧 HELPER FUNCTIONS (Geometry & Cleaning)
# ==========================================

def order_points(pts):
    """Orders coordinates: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Transforms a skewed image into a flat, top-down view."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def remove_shadows(img):
    """Normalizes lighting to remove shadows/glare."""
    rgb_planes = cv2.split(img)
    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(norm_img)
    return cv2.merge(result_planes)

def crop_to_text_content(bin_img, pad=10):
    """Crops white space around the text."""
    inverted = cv2.bitwise_not(bin_img)
    coords = cv2.findNonZero(inverted)
    if coords is None: return bin_img
    
    x, y, w, h = cv2.boundingRect(coords)
    h_img, w_img = bin_img.shape
    
    x_new = max(0, x - pad)
    y_new = max(0, y - pad)
    w_new = min(w_img - x_new, w + 2*pad)
    h_new = min(h_img - y_new, h + 2*pad)
    
    return bin_img[y_new:y_new+h_new, x_new:x_new+w_new]

# ==========================================
# 🚀 MAIN PREPROCESSOR
# ==========================================

def preprocess_image_for_ocr(image_contents: bytes) -> tuple:
    # --- 1. Load Image ---
    np_img = np.frombuffer(image_contents, np.uint8)
    original_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if original_img is None: raise ValueError("Could not decode")

    # --- 2. Receipt Detection (Crop) ---
    # (Keep your existing cropping logic here - it is working fine)
    # ... [PASTE YOUR EXISTING CROP LOGIC HERE] ...
    # For this snippet, I assume 'cropped_img' is ready:
    cropped_img = original_img # <-- REPLACE THIS WITH YOUR CROP LOGIC

    # --- 3. Resize ---
    h, w = cropped_img.shape[:2]
    target_width = 1600
    if w < target_width:
        scale = target_width / w
        cropped_img = cv2.resize(cropped_img, (target_width, int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # --- 4. Clean Shadows (Aggressive) ---
    no_shadow = remove_shadows(cropped_img)
    gray = cv2.cvtColor(no_shadow, cv2.COLOR_BGR2GRAY)

    # --- 5. Binarize (The Fix for "HOON ON Oe") ---
    
    # A. Median Blur: excellent for removing salt-and-pepper noise (watermark dots)
    blurred = cv2.medianBlur(gray, 3) 
    
    # B. Adaptive Threshold with HIGHER Constant (C)
    # Block Size 31, C=25 (Was 10). 
    # This forces light-gray watermarks to become white (ignored), keeping only dark text.
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 25
    )

    # --- 6. Morphological Cleanup ---
    # Remove any remaining tiny dots
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # --- 7. Clean Borders & Crop ---
    h_t, w_t = thresh_clean.shape
    cv2.rectangle(thresh_clean, (0,0), (w_t, h_t), (255,255,255), 20) 
    final_img = crop_to_text_content(thresh_clean, pad=20)

    return final_img, cropped_img



# --- 🔥 DIPERBARUI: Fungsi "asisten" sync sekarang menerima 'category' ---
def process_and_save_sync(image_contents: bytes, filename: str, userId: str, projectId: str, category: str):
    
    tesseract_cmd_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    tessdata_path = r'C:\Program Files\Tesseract-OCR\tessdata'
    temp_image_path = None 

    try:
        # --- 1. PREPROCESSING (The New Pipeline) ---
        # Get the binarized image (final_img) AND the color crop (for potential future use)
        final_img, _ = preprocess_image_for_ocr(image_contents)
        cv2.imwrite('debug_cropped_img.png', _)  # Debugging line to save the preprocessed image
        cv2.imwrite('debug_final_img.png', final_img)  # Debugging line to save the preprocessed image
        processed_pil = Image.fromarray(final_img)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image_file:
            temp_image_path = temp_image_file.name
            processed_pil.save(temp_image_path)

        # --- 2. TESSERACT CONFIGURATION (OPTIMIZED) ---
        final_lang = 'ind+eng'
        
        cmd = [
            tesseract_cmd_path, 
            temp_image_path, 
            "stdout", 
            "-l", final_lang, 
            "--psm", "4",  # Single Column Variable
            "-c", "preserve_interword_spaces=1", 
            "-c", "tessedit_do_invert=0",
            
            # --- NEW: NOISE REDUCTION ---
            # Block typical noise characters often found in separators
            "-c", "tessedit_char_blacklist=|[]{}«»~_—" 
        ]
        
        env = os.environ.copy()
        env['TESSDATA_PREFIX'] = tessdata_path
        
        # Run Tesseract
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, encoding='utf-8', env=env 
        )
        text = result.stdout
        print("OCR Text Output:"+ text)
        # Cleanup temp file immediately
        if temp_image_path:
            os.remove(temp_image_path)
            temp_image_path = None 
        
        if not text.strip():
            text = "Tidak ada teks terdeteksi."

        # --- 3. PARSING ---
        lines = text.splitlines()
        parsed_data = parse_receipt_with_regex_v4_0(text) 

        # --- 4. DATA PREPARATION ---
        result_data = {
            "filename": filename,
            "merchant": parsed_data.get("merchant"),
            "date": parsed_data.get("date"),
            "total": parsed_data.get("total_int"),
            "total_string": parsed_data.get("total_str"),
            "tunai": parsed_data.get("tunai_int"),
            "kembali": parsed_data.get("kembali_int"),
            "tunai_string": parsed_data.get("tunai_str"),
            "kembali_string": parsed_data.get("kembali_str"),
            "items": parsed_data.get("items"),
            # "raw_lines": lines, # Optional: Comment out to save DB space
            "raw_text": text,   # Optional: Comment out to save DB space
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "uploadedBy": userId,
            "category": category
        }
        
        # --- 5. FIREBASE SAVE ---
        doc_ref = db.collection("projects").document(projectId).collection("expenses").document()
        doc_ref.set(result_data) 
        
        # Clean for response
        result_data.pop("raw_lines", None)
        # result_data.pop("raw_text", None)
        result_data["timestamp"] = result_data["timestamp"].isoformat()

        return result_data

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise e 
    
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

@app.post("/extract-text/", response_class=JSONResponse)
async def extract_text_and_save( 
    image: UploadFile = File(...),
    userId: str = Form(...),
    projectId: str = Form(...),
    category: str = Form(...) # <-- 🔥 TAMBAHKAN PARAMETER KATEGORI
):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah bukan gambar.")
    if not userId or not projectId or not category:
        raise HTTPException(status_code=400, detail="userId, projectId, dan category tidak boleh kosong.")

    try:
        contents = await image.read()

        result = await run_in_threadpool(
            process_and_save_sync, 
            image_contents=contents,
            filename=image.filename,
            userId=userId,
            projectId=projectId,
            category=category # <-- 🔥 KIRIM KATEGORI KE FUNGSI SYNC
        )
        
        return JSONResponse(content={"message": f"Struk berhasil diproses untuk project {projectId}", "data": result})

    except Exception as e:
        # ... (error handling-mu) ...
        import traceback
        print(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Terjadi error internal: {str(e)}")


# ============================================================
# 💡 FUNGSI HELPER (Parsing, dll)
# ============================================================
# ... (Semua fungsi helper-mu: MONTH_MAP, clean_price, parse_receipt_with_regex_v4_0) ...
# ... (Tidak perlu diubah) ...
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
