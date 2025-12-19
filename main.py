from io import BytesIO
import os
import re
import datetime
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from functools import lru_cache
import tempfile 
import calendar
import cv2
import numpy as np
from ultralytics import YOLO
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
        print(f"Gagal koneksi ke Firebase. Pastikan file serviceAccountKey.json ada di path: {cred_path}")
        print(f"Error: {e}")
        
db = firestore.client() 

# --- FastAPI Setup ---
app = FastAPI(
    title="OCR Receipt Parser - Project Based",
    version="9.0.0 (Manual Category)" # Versi baru
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# --- Bypass model host connectivity check (may speed startup) ---
# To override externally, set DISABLE_MODEL_SOURCE_CHECK in your environment first.
os.environ.setdefault('DISABLE_MODEL_SOURCE_CHECK', 'True')

@app.get("/")
def read_root():
    return {"message": "Welcome to Project-Based OCR API (Manual Category)!"}

# ==========================================
# 🚀 MAIN PREPROCESSOR
# ==========================================

@lru_cache(maxsize=1)
def load_yolo_model():
    return YOLO("dataset/runs/detect/train3/weights/best.pt")

@lru_cache(maxsize=1)
def load_ocr_model():
    """Load and cache PaddleOCR model"""
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False, 
        use_textline_orientation=False
    )
    return ocr


def extract_text_from_json(result_json: dict):
    """Extract text lines from OCR result JSON"""
    rec_texts = result_json.get("res", {}).get('rec_texts', [])
    if isinstance(rec_texts, list):
        return rec_texts
    return []


def merge_split_lines(text: str) -> str:
    """
    Intelligently merge lines that OCR incorrectly split.
    Handles cases like:
    - TUNAI: \n 100,000 -> TUNAI: 100,000
    - Product name \n qty price total -> Product name qty price total
    """
    lines = text.split('\n')
    merged_lines = []
    i = 0
    
    # Keywords that indicate this is NOT a product line
    NON_ITEM_KEYWORDS = [
        'TOTAL', 'TUNAI', 'KEMBALI', 'CUKAI', 'HARGA', 'BELANJA',
        'LAYANAN', 'KONSUMEN', 'TELP', 'SMS', 'WA', 'SMSWA', 'KONTAK',
        'EMAIL', 'WEBSITE', 'GRATIS', 'ONGKIR', 'KLIK', 'JAM', 
        'TERIMA', 'KASIH', 'MUDAH', 'JUAL'
    ]
    
    while i < len(lines):
        current = lines[i].strip()
        
        # Skip empty lines
        if not current:
            i += 1
            continue
        
        # Look ahead if there's a next line
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            
            # ===== PATTERN 1: TOTAL/TUNAI/KEMBALI followed by amount =====
            # Matches: "TUNAI:", "TOTAL BELANJA :", etc. followed by numbers
            if re.match(r'^(TUNAI|KEMBALI|TOTAL|HARGA\s*JUAL|TOTAL\s*BELANJA)\s*:?\s*$', 
                       current, re.IGNORECASE):
                if next_line and re.match(r'^[\d,.\s]+$', next_line):
                    merged_lines.append(f"{current} {next_line}")
                    i += 2
                    continue
            
            # ===== PATTERN 2: Product name followed by numbers =====
            # Only merge if:
            # - Current line has letters
            # - Next line is only numbers
            # - NOT a metadata line (date, transaction ID)
            # - NOT a non-item keyword
            if (re.search(r'[A-Z]', current) and 
            len(current) > 5 and 
            not any(kw in current.upper() for kw in NON_ITEM_KEYWORDS) and
            not re.search(r'\d{2}\.\d{2}\.\d{2}', current) and
            not re.search(r'[A-Z]\d+-\d+', current) and
            not re.search(r'/[A-Z]+/\d+', current)):
            
                # Check if next 2 lines are numbers (qty price total pattern)
                if (i + 2 < len(lines) and 
                    re.match(r'^[\d,.\s]+$', lines[i + 1].strip()) and 
                    re.match(r'^[\d,.\s]+$', lines[i + 2].strip())):
                    merged_lines.append(f"{current} {lines[i + 1].strip()} {lines[i + 2].strip()}")
                    i += 3
                    continue
        
        # No merge needed, add current line as-is
        merged_lines.append(current)
        i += 1
    
    return '\n'.join(merged_lines)


def fix_spacing_issues(text: str) -> str:
    """
    Fix OCR spacing mistakes while preserving line breaks.
    Processes each line individually to maintain structure.
    """
    lines = text.split('\n')
    fixed_lines = []
    
    for line in lines:
        original_line = line
        
        # === STEP 1: Protect patterns that should NOT be modified ===
        # Protect emails
        line = re.sub(r'(\S+@\S+\.\S+)', 
                     lambda m: m.group(1).replace('@', '<!AT!>').replace('.', '<!DOT!>'), 
                     line)
        
        # Protect phone numbers (0811.1500.280, 08111500280, etc.)
        line = re.sub(r'\b(\d{4}[\.\-]?\d{4}[\.\-]?\d{3,4})\b', 
                     lambda m: m.group(1).replace('.', '<!DOT!>').replace('-', '<!DASH!>'), 
                     line)
        
        # Protect reference numbers (NO.37, 4.0.26, JL.KELINCI)
        line = re.sub(r'\b([A-Z]{2,}\.\d+)\b', 
                     lambda m: m.group(1).replace('.', '<!DOT!>'), 
                     line)
        line = re.sub(r'\b(\d+\.\d+\.\d+)\b', 
                     lambda m: m.group(1).replace('.', '<!DOT!>'), 
                     line)
        
        # === STEP 2: Fix spacing issues ===
        # Add space between letter and number (Item123 -> Item 123)
        line = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', line)
        
        # Add space between number and letter for 2+ digit numbers (123Item -> 123 Item)
        line = re.sub(r'(\d{2,})([A-Z][a-z])', r'\1 \2', line)
        
        # Add space after punctuation if missing (Item,Price -> Item, Price)
        line = re.sub(r'([,;:])([A-Za-z])', r'\1 \2', line)
        
        # Fix merged quantity + price (238000 -> 2 38000)
        # Only split if: digit(s) followed by 4-6 digit price
        line = re.sub(r'\b([1-9])(\d{4,6})\b', r'\1 \2', line)
        line = re.sub(r'\b([1-9]\d)(\d{4,6})\b', r'\1 \2', line)
        
        # Remove multiple spaces (but NOT newlines!)
        line = re.sub(r' +', ' ', line)
        
        # === STEP 3: Restore protected patterns ===
        line = line.replace('<!AT!>', '@')
        line = line.replace('<!DOT!>', '.')
        line = line.replace('<!DASH!>', '-')
        
        fixed_lines.append(line.strip())
    
    return '\n'.join(fixed_lines)

def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Perspective transform to straighten receipt"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([[0, 0], [maxWidth - 1, 0], 
                    [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def preprocess_receipt_image(image_pil):
    """Detect and straighten receipt before OCR"""
    import numpy as np
    import cv2
    
    # Convert PIL to OpenCV
    img = np.array(image_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Resize for faster processing
    height, width = img.shape[:2]
    scale = 1.0
    if width > 800:
        scale = 800 / width
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    # Find receipt contour
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Find rectangular receipt
    receipt_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            receipt_contour = approx
            break
    
    # Apply perspective transform if receipt found
    if receipt_contour is not None:
        receipt_contour = receipt_contour.reshape(4, 2) / scale
        # Convert back to original image
        img_original = np.array(image_pil)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
        warped = four_point_transform(img_original, receipt_contour)
        # Convert back to PIL
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    
    return image_pil

def detect_and_crop_receipt(image_pil):
    model = load_yolo_model()

    img = np.array(image_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = model(img, conf=0.25)[0]

    if not results.boxes:
        return image_pil  # fallback if YOLO fails

    # take the largest box (receipt)
    boxes = results.boxes.xyxy.cpu().numpy()
    areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes]
    x1,y1,x2,y2 = boxes[int(np.argmax(areas))].astype(int)

    crop = img[y1:y2, x1:x2]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    cv2.imwrite("tesyolo.png", crop)
    return Image.fromarray(crop)

def detect_and_crop_all_by_class(image_pil):
    model = load_yolo_model()

    img = np.array(image_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = model(img, conf=0.25)[0]

    outputs = {}

    if not results.boxes:
        return outputs

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    for box, cls in zip(boxes, classes):
        x1,y1,x2,y2 = box.astype(int)
        crop = img[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        outputs.setdefault(cls, []).append(Image.fromarray(crop))

    return outputs

def ocr_pil_images(pil_images, ocr_engine):
    texts = []

    for img in pil_images:
        img_np = np.array(img)  # RGB numpy
        result = ocr_engine.predict(img_np)
        for i, res in enumerate(result):
            result_json = res.json
            text_lines = extract_text_from_json(result_json)
            
            print(f"DEBUG - OCR result {i}: {len(text_lines)} lines detected")
            
            if isinstance(text_lines, list):
                texts.extend(text_lines)
            else:
                texts.append(str(text_lines))
    print(texts)
    return texts

def parse_merchant(info_texts):
    for line in info_texts:
        line = line.strip()
        if len(line) > 3 and any(c.isalpha() for c in line):
            return line
    return None

def parse_total(total_texts):
    for line in total_texts:
        m = re.search(r'([\d]{1,3}(?:[.,]\d{3})+)', line)
        if m:
            return clean_price(m.group(1))
    return 0

def group_item_blocks(lines):
    items = []
    i = 0

    while i < len(lines):
        name = lines[i].strip()
        if not name or re.fullmatch(r'[\d.,Xx×]+', name):
            i += 1
            continue

        qty = 1
        unit_price = None
        total_price = None

        # unit price
        if i+1 < len(lines) and re.fullmatch(r'[\d.,]+', lines[i+1]):
            unit_price = clean_price(lines[i+1])
            i += 1

        # qty marker (X1, ×1)
        if i+1 < len(lines) and re.fullmatch(r'[Xx×]\s*\d+', lines[i+1]):
            qty = int(re.sub(r'\D', '', lines[i+1]))
            i += 1

        # total price
        if i+1 < len(lines) and re.fullmatch(r'[\d.,]+', lines[i+1]):
            total_price = clean_price(lines[i+1])
            i += 1

        if total_price is None and unit_price is not None:
            total_price = unit_price * qty

        items.append({
            "name": name,
            "qty": qty,
            "unit_price": unit_price,
            "price": total_price
        })

        i += 1

    return items


def normalize_item_lines(lines):
    merged = []
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # line is price-only
        if re.fullmatch(r'[\d.,]+', line):
            buffer += " " + line
            merged.append(buffer.strip())
            buffer = ""
        else:
            if buffer:
                merged.append(buffer.strip())
            buffer = line

    if buffer:
        merged.append(buffer.strip())

    return merged

def parse_items(item_texts):
    return group_item_blocks(item_texts)

CLASS_MAP = {
    "info_toko": 0,
    "item_belanja": 1,
    "struk_belanja": 2,
    "total": 3
}

def process_and_save_sync(image_contents, filename: str, userId: str, projectId: str, category: str):
    """
    Main OCR processing function.
    Steps:
    1. Save image temporarily
    2. Run OCR
    3. Merge split lines
    4. Fix spacing
    5. Parse receipt data
    6. Save to Firebase
    """
    temp_image_path = None 
    ocr_engine = load_ocr_model()
    struk = detect_and_crop_receipt(image_contents)
    regions = detect_and_crop_all_by_class(image_contents)


    info_imgs = regions.get(CLASS_MAP["info_toko"], [])
    item_imgs = regions.get(CLASS_MAP["item_belanja"], [])
    total_imgs = regions.get(CLASS_MAP["total"], [])
    if info_imgs:
        info_imgs[0].save("info.jpg")

    if item_imgs:
        item_imgs[0].save("item.jpg")

    if total_imgs:
        total_imgs[0].save("total.jpg")
    # Save PIL Image to temp file
    
    try:
        # === STEP 1: OCR INFERENCE ===
        # result = []
        if ocr_engine is None:
            raise RuntimeError("PaddleOCR not available")
        
        info_texts  = ocr_pil_images(info_imgs, ocr_engine)
        item_texts  = ocr_pil_images(item_imgs, ocr_engine)
        total_texts = ocr_pil_images(total_imgs, ocr_engine)
        # text = ocr_pil_images(struk,ocr_engine)

        merchant = parse_merchant(info_texts)
        items    = parse_items(item_texts)
        total    = parse_total(total_texts)
        print("="*50)
        print("OCR Text Output:")
        # print(text)
        print("="*50)
    
        
        # === STEP 5: PARSE RECEIPT ===
        # You need to provide your parse_receipt_with_regex_v4_0 function
        
        print(f"DEBUG - Parsed merchant: {merchant}")
        print(f"DEBUG - Parsed total: {total}")
        print(f"DEBUG - Parsed items: {len(items)}")
        if int(total) == 0 and items:
            total_int = sum(i["price"] for i in items if i.get("price"))
            total = str(total_int)

        # # === STEP 6: PREPARE DATA ===
        result_data = {
            "filename": filename,
            "merchant": merchant,
            "total": int(total),
            "total_string": total,
            "items": items,
            "raw_text": f"Info: {info_texts}\nitems: {item_texts}\ntotal: {total_texts}\n",
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "uploadedBy": userId,
            "category": category
        }
        
        # # === STEP 7: SAVE TO FIREBASE ===
        doc_ref = db.collection("projects").document(projectId).collection("expenses").document()
        doc_ref.set(result_data)
        
        # Clean for response
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
    category: str = Form(...) # <-- 🔥 TAMBAHKAN PARAMETER KATEGORI
):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah bukan gambar.")
    if not userId or not projectId or not category:
        raise HTTPException(status_code=400, detail="userId, projectId, dan category tidak boleh kosong.")

    try:
        contents = Image.open(BytesIO(await image.read()))
        contents = ImageOps.exif_transpose(contents)
        contents.thumbnail((2000, 2000), Image.LANCZOS)
        contents = contents.convert("RGB")
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
    total_pattern = re.compile(r'(?i)(To[tTl][aA4]l?|TWAL|JUMLAH|TAGIHAN|HARGA\s*JUAL|TOTAL\s*BELANJA|GRAND\s*TOTAL)\s+[:\->\s|«;]*\s*([\d.,\s]+)')
    
    # UPDATED TUNAI PATTERN:
    # Matches: TUNAI, TUMAI, TUHAI, CASH
    tunai_pattern = re.compile(r'(?i)(TU[NMH]AI|CASH|BAYAR|DIBAYAR)\s*[:\->\s|«;]*\s*([\d.,\s]+)')
    kembali_pattern = re.compile(r'(?i)(KEMBALI|KEMBAL|KEMBA|CHANGE)\s*[:\->\s|«;]*\s*([\d.,\s]+)')

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

    FILTER_KEYWORDS = [
        "TOTAL", "TUNAI", "KEMBALI", "PPN", "TAX", "SUBTOTAL", "DISKON", "NPWP", 
        "HARGA JUAL", "ITEM", "CASHIER", "ADMINISTRATOR",
        # Contact & Service info
        "LAYANAN", "KONSUMEN", "TELP", "SMS", "WA", "SMSWA", "CONTACT", "KONTAK",
        "EMAIL", "@", "WEBSITE", "WWW", "HTTP",
        # Promotional text
        "GRATIS", "ONGKIR", "BELANJA", "KLIK", "JAM SAMPAI", "TERIMA KASIH", 
        "THANK YOU", "SELAMAT", "MUDAH",
        # Receipt metadata
        "CUKAI", "NO #", "STRUK", "RECEIPT", "TANGGAL", "DATE"
    ]
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
        if re.search(r'\d{4}[\.\-]?\d{4}[\.\-]?\d{3,4}', line):  # Phone pattern
            continue
        if '@' in line or 'WWW' in line.upper() or 'HTTP' in line.upper():  # Email/URL
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
