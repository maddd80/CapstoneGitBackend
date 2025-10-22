import pytesseract
from PIL import Image
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(
    title="API Ekstraksi Teks dari Gambar (OCR)",
    description="Unggah gambar struk atau invoice untuk mengekstrak teks di dalamnya.",
    version="1.0.0",
)

@app.get("/")
def read_root():
    """Endpoint utama untuk menyapa pengguna."""
    return {"message": "Selamat datang di API OCR! Buka /docs untuk mencoba."}


@app.post("/extract-text/", response_class=JSONResponse)
async def extract_text_from_image(image: UploadFile = File(...)):
    """
    Endpoint untuk menerima file gambar dan mengembalikan teks yang diekstrak.
    """
    # Validasi tipe file (opsional tapi direkomendasikan)
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah bukan gambar.")

    try:
        # Baca konten file yang diunggah ke dalam memori
        contents = await image.read()
        
        # Buka gambar dari data bytes di memori menggunakan Pillow
        img = Image.open(io.BytesIO(contents))

        # Proses OCR menggunakan Pytesseract
        text = pytesseract.image_to_string(img, lang='ind+eng')

        # Jika tidak ada teks yang terdeteksi, berikan pesan yang jelas
        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "filename": image.filename,
                    "message": "Tidak ada teks yang dapat dideteksi pada gambar."
                }
            )

        # Kembalikan hasil dalam format JSON
        return {
            "filename": image.filename,
            "content_type": image.content_type,
            "extracted_text": text
        }

    except Exception as e:
        # Tangani error yang mungkin terjadi selama proses
        raise HTTPException(status_code=500, detail=f"Terjadi error: {str(e)}")