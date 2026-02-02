"""
core/parsers.py — Parsers multi-format : PDF, DOCX, TXT/MD, CSV.

Chaque parser retourne une liste de dicts : {"texte": str, "source": str, "page": int}
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedPage:
    texte: str
    source: str
    page: int


# Extensions supportées et leur parser associé
_EXTENSIONS = {
    ".pdf": "_parser_pdf",
    ".docx": "_parser_docx",
    ".txt": "_parser_texte",
    ".md": "_parser_texte",
    ".csv": "_parser_csv",
}


def extensions_supportees() -> list[str]:
    """Retourne la liste des extensions de fichiers acceptées."""
    return list(_EXTENSIONS.keys())


def parser_document(chemin: Path) -> list[ParsedPage]:
    """
    Factory : parse un document selon son extension.
    Retourne une liste de ParsedPage.
    """
    chemin = Path(chemin)
    ext = chemin.suffix.lower()

    if ext not in _EXTENSIONS:
        raise ValueError(
            f"Format non supporté : {ext}. "
            f"Formats acceptés : {', '.join(extensions_supportees())}"
        )

    parser_fn = globals()[_EXTENSIONS[ext]]
    return parser_fn(chemin)


# --- Parsers spécifiques ---


def _parser_pdf(chemin: Path) -> list[ParsedPage]:
    """Extraction PDF via PyPDF2, avec fallback OCR (pytesseract)."""
    from PyPDF2 import PdfReader

    pages = []
    try:
        lecteur = PdfReader(str(chemin))
    except Exception as e:
        print(f"  Impossible de lire {chemin.name} : {e}")
        return pages

    for num_page, page in enumerate(lecteur.pages, start=1):
        texte = ""
        try:
            texte = page.extract_text() or ""
        except Exception:
            pass

        # Si le texte extrait est trop court, tenter l'OCR
        if len(texte.strip()) < 50:
            texte_ocr = _ocr_page(chemin, num_page)
            if texte_ocr:
                texte = texte_ocr

        if texte.strip():
            pages.append(ParsedPage(
                texte=texte.strip(),
                source=chemin.name,
                page=num_page,
            ))

    return pages


def _ocr_page(chemin: Path, num_page: int) -> str:
    """Tente l'OCR sur une page PDF via pdf2image + pytesseract."""
    try:
        from pdf2image import convert_from_path
        import pytesseract

        images = convert_from_path(
            str(chemin),
            first_page=num_page,
            last_page=num_page,
            dpi=200,
        )
        if images:
            return pytesseract.image_to_string(images[0], lang="fra")
    except ImportError:
        pass
    except Exception:
        pass
    return ""


def _parser_docx(chemin: Path) -> list[ParsedPage]:
    """Extraction DOCX via python-docx (paragraphes)."""
    from docx import Document

    doc = Document(str(chemin))
    texte_complet = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    if not texte_complet.strip():
        return []

    return [ParsedPage(
        texte=texte_complet.strip(),
        source=chemin.name,
        page=1,
    )]


def _parser_texte(chemin: Path) -> list[ParsedPage]:
    """Lecture simple de fichiers TXT/MD."""
    texte = chemin.read_text(encoding="utf-8", errors="ignore")

    if not texte.strip():
        return []

    return [ParsedPage(
        texte=texte.strip(),
        source=chemin.name,
        page=1,
    )]


def _parser_csv(chemin: Path) -> list[ParsedPage]:
    """Conversion CSV en texte via pandas."""
    import pandas as pd

    df = pd.read_csv(str(chemin))
    texte = df.to_string(index=False)

    if not texte.strip():
        return []

    return [ParsedPage(
        texte=texte.strip(),
        source=chemin.name,
        page=1,
    )]
