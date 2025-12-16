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


# --- 🔥 DIPERBARUI: Fungsi "asisten" sync sekarang menerima 'category' ---
def process_and_save_sync(image_contents: bytes, filename: str, userId: str, projectId: str, category: str): # <-- TAMBAH 'category'
    
    tesseract_cmd_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    tessdata_path = r'C:\Program Files\Tesseract-OCR\tessdata'
    temp_image_path = None 

    try:
        # --- 1. PREPROCESSING ("MATA") ---
        # ... (Kode preprocessing-mu tidak diubah) ...
        np_img = np.frombuffer(image_contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        gray_scaled = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
        ret, gray_final = cv2.threshold(gray_scaled, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        processed_pil = Image.fromarray(gray_final)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image_file:
            temp_image_path = temp_image_file.name
            processed_pil.save(temp_image_path)


        # --- 2. TESSERACT VIA SUBPROCESS ---
        # ... (Kode Tesseract-mu tidak diubah) ...
        final_lang = 'ind+eng'
        cmd = [ tesseract_cmd_path, temp_image_path, "stdout", "-l", final_lang, "--psm", "6" ]
        env = os.environ.copy()
        env['TESSDATA_PREFIX'] = tessdata_path
        print(f"Mencoba OCR dengan Subprocess: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, encoding='utf-8', env=env 
        )
        text = result.stdout
        os.remove(temp_image_path)
        temp_image_path = None 
        
        if not text.strip():
            text = "Tidak ada teks terdeteksi." # Jangan error, kirim pesan saja
        
        # ... (Kode print debug-mu) ...

        # --- 3. PARSING ("OTAK") ---
        lines = text.splitlines()
        parsed_data = parse_receipt_with_regex_v4_0(text) 

        # --- 🔥 DIPERBARUI: 4. SIAPKAN DATA HASIL ---
        # HAPUS BLOK MENEBAK KATEGORI
        
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
            "timestamp": datetime.datetime.now(datetime.timezone.utc), # Pakai datetime object
            "uploadedBy": userId,
            "category": category # <-- 🔥 GUNAKAN KATEGORI DARI PARAMETER
        }
        
        # --- 5. SIMPAN KE FIREBASE ---
        doc_ref = db.collection("projects").document(projectId).collection("expenses").document()
        doc_ref.set(result_data) 
        
        # Kirim balik data yang bersih (tanpa text mentah)
        result_data.pop("raw_lines", None)
        result_data.pop("raw_text", None)
        result_data["timestamp"] = result_data["timestamp"].isoformat() # Ubah ke string untuk JSON

        return result_data

    except Exception as e:
        # ... (error handling) ...
        import traceback
        print(traceback.format_exc())
        raise e 
    
    finally:
        # ... (finally block) ...
        if temp_image_path and os.path.exists(temp_image_path):
            print(f"Membersihkan file sementara: {temp_image_path}")
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
    
    total_pattern = re.compile(r'(?i)(Tota[lI]?|TOTAL|TOTAL\s*BELANJA|GRAND\s*TOTAL|TAGIHAN)\s*[:\->\s|«;]*\s*([\d.,\s]+)')
    tunai_pattern = re.compile(r'(?i)(TUNAI|CASH|PEMBAYARAN\s*TUNAI)\s*[:\->\s|«;]*\s*([\d.,\s]+)')
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