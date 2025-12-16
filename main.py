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
    title="OCR Receipt Parser - Enhanced Preprocessing",
    version="10.0.0 (Enhanced Image Processing)"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def read_root():
    return {"message": "Welcome to Enhanced OCR API with Advanced Preprocessing!"}


# ============================================================
# 🔥 SIMPLIFIED PREPROCESSING FUNCTIONS
# ============================================================

def preprocess_invoice_image(image_contents: bytes, save_debug=False):
    """
    Simplified preprocessing pipeline:
    1. Grayscale
    2. Resize (max 2.0x)
    3. CLAHE ringan
    4. Adaptive threshold
    5. Morphological close ringan
    """
    
    print("\n" + "="*50)
    print("🖼️  MEMULAI PREPROCESSING IMAGE")
    print("="*50)
    
    # 1. Load & convert to grayscale
    np_img = np.frombuffer(image_contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Gagal membaca gambar")
    
    print(f"✓ Original size: {img.shape[1]}x{img.shape[0]}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize (max 2.0x)
    h, w = gray.shape
    
    if h < 1000:
        scale = 2.0
    elif h < 1500:
        scale = 1.5
    else:
        scale = 1.0
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    if scale > 1.0:
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"✓ Resized to: {new_w}x{new_h} (scale: {scale:.1f}x)")
    else:
        print(f"✓ No resize needed ({w}x{h})")
    
    # 3. CLAHE ringan
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    print("✓ CLAHE applied (light)")
    
    # 4. Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    print("✓ Adaptive threshold applied")
    
    # 5. Morphological close ringan
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    print("✓ Morphological close applied (light)")
    
    # Optional: Save debug images
    if save_debug:
        try:
            cv2.imwrite("debug_1_original.png", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            cv2.imwrite("debug_2_after_clahe.png", gray)
            cv2.imwrite("debug_3_final.png", binary)
            print("✓ Debug images saved")
        except Exception as e:
            print(f"⚠ Debug save failed: {e}")
    
    print("="*50)
    print("✅ PREPROCESSING SELESAI")
    print("="*50 + "\n")
    
    return binary


# ============================================================
# 🔥 UPDATED: Main Processing Function
# ============================================================

def process_and_save_sync(image_contents: bytes, filename: str, userId: str, projectId: str, category: str):
    
    tesseract_cmd_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    tessdata_path = r'C:\Program Files\Tesseract-OCR\tessdata'
    temp_image_path = None 

    try:
        # --- 1. ENHANCED PREPROCESSING ---
        processed_img = preprocess_invoice_image(image_contents, save_debug=True)
        
        # Convert to PIL and save to temp file
        processed_pil = Image.fromarray(processed_img)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image_file:
            temp_image_path = temp_image_file.name
            processed_pil.save(temp_image_path, dpi=(300, 300))  # Set DPI tinggi
            print(f"✓ Temp image saved: {temp_image_path}")

        # --- 2. TESSERACT VIA SUBPROCESS ---
        # Untuk printed receipt (bukan thermal), gunakan settings berbeda
        final_lang = 'eng+deu'  # English + German untuk receipt Swiss
        
        # PSM 6 untuk receipt standar (uniform text block)
        cmd = [
            tesseract_cmd_path, 
            temp_image_path, 
            "stdout", 
            "-l", final_lang,
            "--psm", "6",  
            "--oem", "1",  # LSTM only
            "-c", "preserve_interword_spaces=1",
            # Whitelist karakter yang mungkin ada di receipt
            "-c", "tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüÄÖÜß .,:-/()@CHFEURRpx"
        ]
        
        env = os.environ.copy()
        env['TESSDATA_PREFIX'] = tessdata_path
        
        print(f"🔍 Running OCR (eng+ind, PSM 6)...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, encoding='utf-8', env=env 
        )
        text = result.stdout
        
        # Jika PSM 6 gagal, coba PSM 4 (single column)
        if len(text.strip()) < 100:
            print("⚠ PSM 6 hasil kurang, coba PSM 4...")
            cmd[6] = "4"  
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, encoding='utf-8', env=env 
            )
            text_psm4 = result.stdout
            
            if len(text_psm4.strip()) > len(text.strip()):
                text = text_psm4
                print("✓ Menggunakan hasil PSM 4")
        
        # Jika masih kurang, coba PSM 3 (fully automatic)
        if len(text.strip()) < 100:
            print("⚠ Masih kurang, coba PSM 3...")
            cmd[6] = "3"
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, encoding='utf-8', env=env 
            )
            text_psm3 = result.stdout
            
            if len(text_psm3.strip()) > len(text.strip()):
                text = text_psm3
                print("✓ Menggunakan hasil PSM 3")
        
        # Cleanup temp file
        os.remove(temp_image_path)
        temp_image_path = None 
        
        if not text.strip():
            text = "Tidak ada teks terdeteksi."
        
        # Post-processing: clean up common OCR errors
        text = text.replace('|', 'l')  # Common error: | → l
        text = text.replace('0O', '00')  # O → 0
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces → single space
        
        print(f"\n📄 OCR Result ({len(text)} chars):")
        print("-" * 50)
        print(text[:1000])
        if len(text) > 1000:
            print("... (truncated)")
        print("-" * 50 + "\n")

        # --- 3. PARSING ---
        lines = text.splitlines()
        parsed_data = parse_receipt_with_regex_v4_0(text) 

        # --- 4. PREPARE RESULT DATA ---
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
            "raw_lines": lines,
            "raw_text": text,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "uploadedBy": userId,
            "category": category
        }
        
        # --- 5. SAVE TO FIREBASE ---
        doc_ref = db.collection("projects").document(projectId).collection("expenses").document()
        doc_ref.set(result_data) 
        print(f"✅ Data saved to Firebase: projects/{projectId}/expenses/{doc_ref.id}")
        
        # Return clean data
        result_data.pop("raw_lines", None)
        result_data.pop("raw_text", None)
        result_data["timestamp"] = result_data["timestamp"].isoformat()

        return result_data

    except Exception as e:
        import traceback
        print("❌ ERROR:")
        print(traceback.format_exc())
        raise e 
    
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            print(f"🧹 Cleaning up: {temp_image_path}")
            os.remove(temp_image_path)


@app.post("/extract-text/", response_class=JSONResponse)
async def extract_text_and_save( 
    image: UploadFile = File(...),
    userId: str = Form(...),
    projectId: str = Form(...),
    category: str = Form(...)
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
            category=category
        )
        
        return JSONResponse(content={
            "message": f"Struk berhasil diproses untuk project {projectId}", 
            "data": result
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Terjadi error internal: {str(e)}")


# ============================================================
# 💡 HELPER FUNCTIONS (Parsing, etc)
# ============================================================

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
    lines = text.splitlines()
    total_int = 0; total_str = None; merchant = None; date = None; items = []
    tunai_int = 0; tunai_str = None; kembali_int = 0; kembali_str = None
    
    # Patterns untuk Indonesia
    total_pattern = re.compile(r'(?i)(Tota[lI]?|TOTAL|TOTAL\s*BELANJA|GRAND\s*TOTAL|TAGIHAN)\s*[:\->\s|«;]*\s*([\d.,\s]+)')
    tunai_pattern = re.compile(r'(?i)(TUNAI|CASH|PEMBAYARAN\s*TUNAI)\s*[:\->\s|«;]*\s*([\d.,\s]+)')
    kembali_pattern = re.compile(r'(?i)(KEMBA[LI]?|CHANGE)\s*[:\->\s|«;]*\s*([\d.,\s]+)')
    
    # Patterns untuk Swiss/European receipts
    total_pattern_chf = re.compile(r'(?i)Total\s*[:]*\s*CHF\s*([\d.,\s]+)')
    
    # Date patterns - tambah format Swiss
    date_pattern_1 = re.compile(r'(\d{2}/\d{2}/\d{4})') 
    date_pattern_2 = re.compile(r'(\d{2}\.\d{2}\.\d{2,4})')
    date_pattern_3 = re.compile(r'(\d{2}-\d{2}-\d{2,4})')
    date_pattern_4 = re.compile(r'(?i)Tangga[l]?\s*:\s*(\S+)') 
    date_pattern_5 = re.compile(r'(\d{2}\.\d{2}\.\d{2})') 
    date_pattern_6 = re.compile(r'(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})')
    date_pattern_swiss = re.compile(r'(\d{2})\.(\d{2})\.(\d{4})/(\d{2}):(\d{2}):(\d{2})')  # Swiss format

    # Item patterns
    item_pattern_1 = re.compile(r'(?i)^(\d+)\s*[xX]\s+(.+?)\s+([\d.,]+)')
    item_pattern_2 = re.compile(r'^([a-zA-Z\s\d/.]+?)\s+(\d+)\s+([\d.,]+)\s+([\d.,]+)')
    item_pattern_3 = re.compile(r'^([a-zA-Z][a-zA-Z\s./-]{3,})\s+([\d.,]{3,})')
    item_pattern_4 = re.compile(r'^(\d+)\s+(.+?)\s+(\d+)\s+([\d.,]+)\s+(?:[\d.,]+\s+)?([\d.,]+)')
    
    # Swiss receipt pattern: "1xItem à 10.50 CHF 10.50"
    item_pattern_swiss = re.compile(r'(\d+)x([A-Za-zäöüÄÖÜß\s]+)\s+à\s+([\d.,]+)\s*CHF\s+([\d.,]+)')

    FILTER_KEYWORDS = ["TOTAL", "TUNAI", "KEMBALI", "PPN", "TAX", "SUBTOTAL", 
                       "DISKON", "NPWP", "HARGA JUAL", "ITEM", "CASHIER", "ADMINISTRATOR",
                       "MWST", "INKL", "ENTSPRICHT"]
    
    # Extract merchant (first line with text)
    for line in lines:
        line_clean = line.strip()
        if not merchant and len(line_clean) > 3 and any(c.isalpha() for c in line_clean):
            if "NO #" not in line_clean.upper() and "RECH" not in line_clean.upper():
                merchant = line_clean
            if any(store in line_clean.lower() for store in ["indomaret", "super indo", "alfamart", "lilac", "berghotel", "restaurant"]):
                merchant = line_clean
                break 

    # Parse lines
    for line in lines:
        line = line.strip()
        if not line: continue 

        # Date parsing - Swiss format first
        if not date:
            match_swiss = date_pattern_swiss.search(line)
            if match_swiss:
                day = match_swiss.group(1)
                month = match_swiss.group(2)
                year = match_swiss.group(3)
                date = f"{day}-{month}-{year}"
                continue
            
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
                except Exception: 
                    pass
            
            match = date_pattern_1.search(line) or date_pattern_2.search(line) or \
                    date_pattern_3.search(line) or date_pattern_4.search(line) or \
                    date_pattern_5.search(line)
            if match and "NO #" not in line.upper():
                date = match.group(1).replace('.', '-').replace('/', '-')
                continue 

        # Total - CHF format first
        total_match_chf = total_pattern_chf.search(line)
        if total_match_chf:
            total_str = total_match_chf.group(1)
            total_int = clean_price(total_str)
            continue
            
        total_match = total_pattern.search(line)
        if total_match: 
            total_str = total_match.group(2)
            total_int = clean_price(total_str)
            continue
            
        tunai_match = tunai_pattern.search(line)
        if tunai_match: 
            tunai_str = tunai_match.group(2)
            tunai_int = clean_price(tunai_str)
            continue
            
        kembali_match = kembali_pattern.search(line)
        if kembali_match: 
            kembali_str = kembali_match.group(2)
            kembali_int = clean_price(kembali_str)
            continue

        # Skip keyword lines
        if any(keyword in line.upper() for keyword in FILTER_KEYWORDS):
            continue 

        # Item parsing - Swiss format first
        match_swiss = item_pattern_swiss.search(line)
        if match_swiss:
            try:
                qty = int(match_swiss.group(1))
                name = match_swiss.group(2).strip()
                unit_price = clean_price(match_swiss.group(3))
                price = clean_price(match_swiss.group(4))
                if price > 0:
                    items.append({"name": name, "qty": qty, "price": price, "unit_price": unit_price})
                continue
            except Exception: 
                pass

        # Try other patterns
        match_4 = item_pattern_4.search(line)
        if match_4:
            try:
                name = match_4.group(2).strip()
                qty = int(match_4.group(3))
                price = clean_price(match_4.group(5))
                unit_price = clean_price(match_4.group(4))
                if price > 0: 
                    items.append({"name": name, "qty": qty, "price": price, "unit_price": unit_price})
                continue 
            except Exception: pass

        match_1 = item_pattern_1.search(line)
        if match_1:
            try:
                qty = int(match_1.group(1))
                name = match_1.group(2).strip()
                price = clean_price(match_1.group(3))
                unit_price = price / qty if qty > 0 else 0
                if price > 0: 
                    items.append({"name": name, "qty": qty, "price": price, "unit_price": unit_price})
                continue 
            except Exception: pass

        match_2 = item_pattern_2.search(line)
        if match_2:
            try:
                name = match_2.group(1).strip()
                qty = int(match_2.group(2))
                unit_price = clean_price(match_2.group(3))
                price = clean_price(match_2.group(4)) 
                if price > 0: 
                    items.append({"name": name, "qty": qty, "price": price, "unit_price": unit_price})
                continue 
            except Exception: pass
            
        match_3 = item_pattern_3.search(line)
        if match_3:
            try:
                name = match_3.group(1).strip()
                price = clean_price(match_3.group(2))
                if price > 0: 
                    items.append({"name": name, "qty": 1, "price": price, "unit_price": price})
                continue 
            except Exception: pass
            
    return {
        "merchant": merchant,
        "date": date,
        "total_str": total_str,
        "total_int": total_int,
        "items": items,
        "tunai_str": tunai_str,
        "tunai_int": tunai_int,
        "kembali_str": kembali_str,
        "kembali_int": kembali_int
    }