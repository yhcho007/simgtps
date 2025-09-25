import pytesseract
from pdfminer.high_level import extract_text
from PIL import Image

def extract_pdf_text(path):
    try:
        return extract_text(path)
    except Exception:
        return ''

def ocr_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))
