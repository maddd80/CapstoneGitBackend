import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
import pytesseract  # Ganti ini dengan library OCR-mu jika beda
import datetime

# --- Import & Inisialisasi Firebase ---
import firebase_admin
from firebase_admin import credentials, firestore

# 1. Inisialisasi Kredensial
# Pastikan file "serviceAccountKey.json" ada di folder yang sama
cred = credentials.Certificate("capstone-ad4dc-firebase-adminsdk-fbsvc-3005a411f0.json")

# 2. Inisialisasi Aplikasi Firebase
# Ganti databaseURL dengan link REALTIME DATABASE kamu (meskipun kita pakai Firestore)
# Ini bug/fitur, kamu tetap harus menyertakannya untuk inisialisasi.
# Ambil link ini dari Firebase Console (Realtime Database, BUKAN Firestore)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://capstone-ad4dc-default-rtdb.firebaseio.com/'
})

# 3. Dapatkan Klien Database Firestore
db = firestore.client()
# ----------------------------------------

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Definisikan ID user (Hardcoded untuk sekarang)
# IDEALNYA: Ini didapat dari token otentikasi yang dikirim Flutter
USER_ID = "user_satu"

@app.post("/extract-text/")
async def extract_text_and_save(image: UploadFile = File(...)):
   
    # --- 1. Proses Gambar & OCR (Bagianmu) ---
    try:
        # Baca file gambar yang di-upload
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="File yang di-upload bukan gambar yang valid")

        # Lakukan OCR
        # Ganti ini dengan logika OCR-mu yang lebih canggih
        raw_text = pytesseract.image_to_string(img)

        if not raw_text:
            raw_text = "Tidak ada teks terdeteksi oleh server."

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat proses OCR: {str(e)}")

   
    # --- 2. (OPSIONAL) Parsing Jumlah ---
    # Ini adalah bagian tersulit. Untuk sekarang, kita simpan 0.
    # Kamu bisa parsing 'raw_text' di sini untuk mencari kata "TOTAL"
    parsed_amount = 0.0


    # --- 3. Simpan ke Firebase ---
    try:
        # Buat referensi ke koleksi 'expenses' milik user
        doc_ref = db.collection('users').document(USER_ID).collection('expenses').document()
       
        # Simpan data
        doc_ref.set({
            'jumlah': parsed_amount,              # Hasil parsing (masih 0)
            'teksOcr': raw_text,                  # Teks mentah dari OCR
            'imageUrl': None,                     # Akan diisi jika kamu upload ke Storage
            'timestamp': datetime.datetime.now(datetime.timezone.utc) # Timestamp server
        })

    except Exception as e:
        # Jika gagal simpan ke Firebase
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan ke Firebase: {str(e)}")

   
    # --- 4. Kirim Balikan Sukses ke Flutter ---
    # Jika sampai di sini, berarti semua sukses
    return {"message": "Struk berhasil diproses dan disimpan oleh backend"}


# Perintah untuk menjalankan server
if __name__ == "__main__":
    # Ganti '0.0.0.0' dengan IP lokalmu jika perlu,
    # tapi '0.0.0.0' bagus agar bisa diakses dari HP-mu
    uvicorn.run(app, host="0.0.0.0", port=8000)