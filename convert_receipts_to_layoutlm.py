import os
import json
import cv2
from paddleocr import PaddleOCR
from shapely.geometry import Polygon, Point
from PIL import Image

# ---------------- CONFIG ----------------

DATASET_ROOT = "dataset"
SPLITS = ["train", "valid", "test"]
OUTPUT_DIR = "layoutlm_data"

LABEL_MAP = {
    "merchant_header": "MERCHANT",
    "item_row": "ITEM",
    "item_price": "ITEM",
    "total": "TOTAL",
    "subtotal": "TOTAL",
    "tax": "TOTAL"
}


ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False, 
        use_textline_orientation=False
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- HELPERS ----------------

def load_regions(txt_path):
    regions = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            coords = list(map(float, parts[:8]))
            label = parts[8]

            polygon = Polygon([
                (coords[0], coords[1]),
                (coords[2], coords[3]),
                (coords[4], coords[5]),
                (coords[6], coords[7]),
            ])

            regions.append({
                "polygon": polygon,
                "label": LABEL_MAP.get(label, "O")
            })
    return regions


def normalize_bbox(bbox, w, h):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return [
        int(1000 * min(xs) / w),
        int(1000 * min(ys) / h),
        int(1000 * max(xs) / w),
        int(1000 * max(ys) / h),
    ]


def assign_label(word_bbox, regions):
    word_poly = Polygon(word_bbox)

    for region in regions:
        if not region["polygon"].intersects(word_poly):
            continue

        inter_area = region["polygon"].intersection(word_poly).area
        if inter_area / word_poly.area > 0.3:
            return region["label"]

    return "O"

# ---------------- MAIN ----------------

for split in SPLITS:
    print(f"Processing {split}...")
    samples = []

    img_dir = os.path.join(DATASET_ROOT, split, "images")
    lbl_dir = os.path.join(DATASET_ROOT, split, "labelTxt")

    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(img_dir, img_name)
        txt_path = os.path.join(lbl_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        if not os.path.exists(txt_path):
            continue

        image = Image.open(img_path).convert("RGB")
        image = image.resize((640, 640), Image.BILINEAR)
        w, h = image.size

        regions = load_regions(txt_path)

        ocr_result = ocr.predict(img_path)

        words = []
        boxes = []
        labels = []

        for res in ocr_result:
            js = res.json
            res_arr = js.get("res", {})
            polys = res_arr.get("dt_polys", [])
            texts = res_arr.get("rec_texts", [])

            for bbox, text in zip(polys, texts):
                text = text.strip()
                if not text:
                    continue

                label = assign_label(bbox, regions)
                norm_box = normalize_bbox(bbox, w, h)

                words.append(text)
                boxes.append(norm_box)
                labels.append(label)

        if words:
            samples.append({
                "id": img_name,
                "words": words,
                "bboxes": boxes,
                "labels": labels
            })

    with open(os.path.join(OUTPUT_DIR, f"{split}.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

print("Conversion finished.")
