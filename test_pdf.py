from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

def extract_text_with_ocr(pdf_path):
    """Extrait le texte avec OCR si nécessaire"""
    
    # Essaie d'abord PyPDF2 (rapide)
    reader = PdfReader(pdf_path)
    print(f"📄 Fichier : {pdf_path}")
    print(f"📄 Nombre de pages : {len(reader.pages)}\n")
    
    for i, page in enumerate(reader.pages):
        print(f"{'='*60}")
        print(f"--- PAGE {i+1} ---")
        print(f"{'='*60}\n")
        
        # Extraction normale
        text = page.extract_text()
        
        # Si le texte est vide ou très court, utilise l'OCR
        if len(text.strip()) < 50:
            print("⚠️  Peu de texte détecté, utilisation de l'OCR...\n")
            
            # Convertit la page en image
            images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
            
            # OCR sur l'image
            text = pytesseract.image_to_string(images[0], lang='fra')
            print("🔍 Texte extrait par OCR :\n")
        else:
            print("✅ Texte extrait normalement :\n")
        
        # Affiche les 800 premiers caractères
        print(text[:800])
        
        if len(text) > 800:
            print(f"\n... ({len(text)} caractères au total)")
        
        print("\n")

# Test sur tous les PDFs
import os

pdf_dir = "./documents"
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_dir, pdf_file)
    print(f"\n{'#'*60}")
    print(f"# {pdf_file}")
    print(f"{'#'*60}\n")
    
    try:
        extract_text_with_ocr(pdf_path)
    except Exception as e:
        print(f"❌ Erreur : {e}\n")
    
    print("\n" + "="*60 + "\n")
