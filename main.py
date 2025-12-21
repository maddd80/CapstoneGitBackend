"""
OCR Receipt Parser API
A FastAPI-based service for extracting structured data from receipt images
using YOLO object detection and PaddleOCR text recognition.
"""

import os
import re
import datetime
import calendar
from io import BytesIO
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from ultralytics import YOLO
from paddleocr import PaddleOCR
import firebase_admin
from firebase_admin import credentials, firestore


# ==========================================
# CONFIGURATION
# ==========================================

FIREBASE_CRED_PATH = "capstone-ad4dc-firebase-adminsdk-fbsvc-3005a411f0.json"
FIREBASE_DATABASE_URL = "https://capstone-ad4dc-default-rtdb.firebaseio.com/"
YOLO_MODEL_PATH = "dataset/runs/detect/train3/weights/best.pt"

# Disable model source check for faster startup
os.environ.setdefault('DISABLE_MODEL_SOURCE_CHECK', 'True')

# Class mapping for YOLO detection
CLASS_MAP = {
    "info_toko": 0,
    "item_belanja": 1,
    "struk_belanja": 2,
    "total": 3
}

# Keywords to filter out from item detection
NON_ITEM_KEYWORDS = [
    'TOTAL', 'TUNAI', 'KEMBALI', 'PPN', 'TAX', 'SUBTOTAL', 'DISKON', 'NPWP',
    'HARGA JUAL', 'ITEM', 'CASHIER', 'ADMINISTRATOR', 'LAYANAN', 'KONSUMEN',
    'TELP', 'SMS', 'WA', 'SMSWA', 'CONTACT', 'KONTAK', 'EMAIL', '@', 'WEBSITE',
    'WWW', 'HTTP', 'GRATIS', 'ONGKIR', 'BELANJA', 'KLIK', 'JAM SAMPAI',
    'TERIMA KASIH', 'THANK YOU', 'SELAMAT', 'MUDAH', 'CUKAI', 'NO #',
    'STRUK', 'RECEIPT', 'TANGGAL', 'DATE'
]

MONTH_MAP = {name.lower(): num for num, name in enumerate(calendar.month_abbr) if num}
MONTH_MAP['mar'] = 3


# ==========================================
# FIREBASE INITIALIZATION
# ==========================================

def initialize_firebase():
    """Initialize Firebase connection"""
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DATABASE_URL})
            print("✅ Firebase initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Firebase: {e}")
            raise

initialize_firebase()
db = firestore.client()


# ==========================================
# FASTAPI SETUP
# ==========================================

app = FastAPI(
    title="OCR Receipt Parser API",
    description="Extract structured data from receipt images",
    version="9.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "message": "OCR Receipt Parser API",
        "status": "running",
        "version": "9.0.0"
    }


# ==========================================
# MODEL LOADING
# ==========================================

@lru_cache(maxsize=1)
def load_yolo_model() -> YOLO:
    """Load and cache YOLO object detection model"""
    return YOLO(YOLO_MODEL_PATH)


@lru_cache(maxsize=1)
def load_ocr_model() -> PaddleOCR:
    """Load and cache PaddleOCR text recognition model"""
    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )


# ==========================================
# IMAGE PROCESSING UTILITIES
# ==========================================

def detect_and_crop_all_by_class(image_pil: Image.Image) -> Dict[int, List[Image.Image]]:
    """Detect and crop all receipt regions by class (info, items, total)"""
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
        x1, y1, x2, y2 = box.astype(int)
        crop = img[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        outputs.setdefault(cls, []).append(Image.fromarray(crop))

    return outputs


# ==========================================
# OCR & TEXT PROCESSING
# ==========================================

def extract_text_from_json(result_json: dict) -> List[str]:
    """Extract text lines from OCR result JSON"""
    rec_texts = result_json.get("res", {}).get('rec_texts', [])
    return rec_texts if isinstance(rec_texts, list) else []


def ocr_pil_images(pil_images: List[Image.Image], ocr_engine: PaddleOCR) -> List[str]:
    """Run OCR on a list of PIL images and return extracted text lines"""
    texts = []
    for img in pil_images:
        img_np = np.array(img)
        result = ocr_engine.predict(img_np)
        
        for res in result:
            result_json = res.json
            text_lines = extract_text_from_json(result_json)
            if isinstance(text_lines, list):
                texts.extend(text_lines)
            else:
                texts.append(str(text_lines))
    
    return texts

#==========================================
# RECEIPT PARSING
# ==========================================

def clean_price(price_str: str) -> int:
    """Extract numeric value from price string"""
    if not price_str:
        return 0
    try:
        cleaned = re.sub(r'\D', '', price_str)
        return int(cleaned)
    except (ValueError, TypeError):
        return 0


def parse_merchant(info_texts: List[str]) -> Optional[str]:
    """Extract merchant name from info texts"""
    for line in info_texts:
        line = line.strip()
        if len(line) > 3 and any(c.isalpha() for c in line):
            return line
    return None


def parse_total(total_texts: List[str]) -> int:
    """Extract total amount from total texts"""
    for line in total_texts:
        match = re.search(r'([\d]{1,3}(?:[.,]\d{3})+)', line)
        if match:
            return clean_price(match.group(1))
    return 0


def group_item_blocks(lines: List[str]) -> List[Dict]:
    """Parse item lines into structured item data"""
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

        # Check for unit price
        if i + 1 < len(lines) and re.fullmatch(r'[\d.,]+', lines[i + 1]):
            unit_price = clean_price(lines[i + 1])
            i += 1

        # Check for quantity marker (X1, ×1)
        if i + 1 < len(lines) and re.fullmatch(r'[Xx×]\s*\d+', lines[i + 1]):
            qty = int(re.sub(r'\D', '', lines[i + 1]))
            i += 1

        # Check for total price
        if i + 1 < len(lines) and re.fullmatch(r'[\d.,]+', lines[i + 1]):
            total_price = clean_price(lines[i + 1])
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


def parse_items(item_texts: List[str]) -> List[Dict]:
    """Parse item texts into structured item list"""
    return group_item_blocks(item_texts)


# ==========================================
# MAIN PROCESSING PIPELINE
# ==========================================

def process_and_save_sync(
    image_contents: Image.Image,
    filename: str,
    userId: str,
    projectId: str,
    category: str
) -> Dict:
    """
    Main OCR processing pipeline:
    1. Detect and crop receipt regions
    2. Run OCR on each region
    3. Parse merchant, items, and total
    4. Save to Firebase
    """
    try:
        ocr_engine = load_ocr_model()
        
        # Detect and crop regions
        regions = detect_and_crop_all_by_class(image_contents)
        info_imgs = regions.get(CLASS_MAP["info_toko"], [])
        item_imgs = regions.get(CLASS_MAP["item_belanja"], [])
        total_imgs = regions.get(CLASS_MAP["total"], [])

        # Run OCR on each region
        info_texts = ocr_pil_images(info_imgs, ocr_engine)
        item_texts = ocr_pil_images(item_imgs, ocr_engine)
        total_texts = ocr_pil_images(total_imgs, ocr_engine)

        # Parse data
        merchant = parse_merchant(info_texts)
        items = parse_items(item_texts)
        total = parse_total(total_texts)

        print("=" * 50)
        print(f"Merchant: {merchant}")
        print(f"Total: {total}")
        print(f"Items: {len(items)}")
        print("=" * 50)

        # Calculate total from items if not detected
        if int(total) == 0 and items:
            total = sum(i["price"] for i in items if i.get("price"))

        # Prepare result data
        result_data = {
            "filename": filename,
            "merchant": merchant,
            "total": int(total),
            "total_string": str(total),
            "items": items,
            "raw_text": f"Info: {info_texts}\nItems: {item_texts}\nTotal: {total_texts}",
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "uploadedBy": userId,
            "category": category
        }

        # Save to Firebase
        doc_ref = db.collection("projects").document(projectId).collection("expenses").document()
        doc_ref.set(result_data)

        # Convert timestamp for JSON response
        result_data["timestamp"] = result_data["timestamp"].isoformat()

        return result_data

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise


# ==========================================
# API ENDPOINTS
# ==========================================

@app.post("/extract-text/", response_class=JSONResponse)
async def extract_text_and_save(
    image: UploadFile = File(...),
    userId: str = Form(...),
    projectId: str = Form(...),
    category: str = Form(...)
):
    """
    Extract text from receipt image and save to Firebase
    
    Args:
        image: Receipt image file
        userId: ID of the user uploading the receipt
        projectId: ID of the project this expense belongs to
        category: Expense category (e.g., "food", "transport")
    
    Returns:
        JSON response with extracted receipt data
    """
    # Validate input
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")
    
    if not userId or not projectId or not category:
        raise HTTPException(
            status_code=400,
            detail="userId, projectId, and category are required"
        )

    try:
        # Load and preprocess image
        contents = Image.open(BytesIO(await image.read()))
        contents = ImageOps.exif_transpose(contents)
        contents.thumbnail((2000, 2000), Image.LANCZOS)
        contents = contents.convert("RGB")

        # Process in thread pool to avoid blocking
        result = await run_in_threadpool(
            process_and_save_sync,
            image_contents=contents,
            filename=image.filename,
            userId=userId,
            projectId=projectId,
            category=category
        )

        return JSONResponse(content={
            "message": f"Receipt processed successfully for project {projectId}",
            "data": result
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 