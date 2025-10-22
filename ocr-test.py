from PIL import Image
import pytesseract
import os

# --- PENTING UNTUK PENGGUNA WINDOWS ---
# Jika Tesseract tidak di-install di lokasi default atau tidak ada di PATH sistem,
# Anda harus menunjukkan lokasinya secara manual di sini.
# Hapus tanda pagar '#' di bawah dan sesuaikan path-nya jika perlu.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text_from_image(image_path):
    """
    Fungsi untuk mengekstrak teks dari sebuah gambar menggunakan Tesseract OCR.

    Args:
        image_path (str): Path menuju file gambar.

    Returns:
        str: Teks yang berhasil diekstrak. Mengembalikan pesan error jika gagal.
    """
    if not os.path.exists(image_path):
        return f"Error: File tidak ditemukan di '{image_path}'"

    try:
        # Buka gambar menggunakan library Pillow
        image = Image.open(image_path)

        # Gunakan pytesseract untuk mengubah gambar menjadi string (teks)
        # Tambahkan 'lang="ind"' jika Anda secara spesifik ingin memproses teks Bahasa Indonesia
        text = pytesseract.image_to_string(image, lang='ind+eng')

        return text
    except Exception as e:
        return f"Terjadi error saat memproses gambar: {e}"

# --- Contoh Penggunaan ---
if __name__ == "__main__":
    # Ganti 'contoh_struk.png' dengan nama file gambar Anda.
    # Pastikan file gambar berada di folder yang sama dengan script Python ini,
    # atau sertakan path lengkapnya.
    image_file = 'contoh_struk.jpg'
    
    # Panggil fungsi untuk ekstraksi
    extracted_text = extract_text_from_image(image_file)

    # Cetak hasilnya
    print("--------------------------------------------------")
    print(f"Hasil Ekstraksi Teks dari: {image_file}")
    print("--------------------------------------------------")
    print(extracted_text)
    print("--------------------------------------------------")
    print("Proses Selesai.")