# ocr_utils.py (Tesseract version)
import os
import pytesseract
from PIL import Image
import io

# Point pytesseract to your Tesseract installation on D:
if os.path.exists("C:/Program Files/Tesseract-OCR/tesseract.exe"):
    pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe" #for windows, change accordingly for macos
elif os.path.exists("/opt/homebrew/bin/tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract" #for macos, change accordingly for windows

def extract_text_from_image(image_bytes: bytes, lang: str = "eng") -> str:
    """
    Extract text from image using Tesseract OCR.

    Args:
        image_bytes (bytes): Raw bytes of the image
        lang (str): Language codes, e.g., "eng", "eng+urd"

    Returns:
        str: Extracted text
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text = pytesseract.image_to_string(image, lang=lang)
        return text.strip()
    except Exception as e:
        print(f"[OCR] Error: {e}")
        return ""
