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
    ".xlsx": "_parser_excel",
    ".xls": "_parser_excel",
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
    """Extraction PDF via PyMuPDF4LLM — meilleure qualité que PyPDF2.

    Retourne le texte en markdown (tableaux, titres, listes préservés).
    """
    import pymupdf4llm

    pages = []
    try:
        pages_md = pymupdf4llm.to_markdown(str(chemin), page_chunks=True)
    except Exception as e:
        print(f"  Impossible de lire {chemin.name} : {e}")
        return pages

    for page_data in pages_md:
        texte = page_data.get("text", "")
        num_page = page_data.get("metadata", {}).get("page", 1)
        # pymupdf4llm utilise un index 0-based pour les pages
        num_page = num_page + 1 if isinstance(num_page, int) else 1

        if texte.strip():
            pages.append(ParsedPage(
                texte=texte.strip(),
                source=chemin.name,
                page=num_page,
            ))

    return pages


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


def _parser_excel(chemin: Path) -> list[ParsedPage]:
    """Extraction Excel (.xlsx et .xls) via pandas.

    Gère les fichiers Excel multi-feuilles en concaténant toutes les feuilles.
    Chaque feuille est préfixée par son nom pour le contexte.

    Formats supportés:
    - .xlsx (Excel 2007+) via openpyxl
    - .xls (Excel 97-2003) via xlrd
    """
    import pandas as pd

    pages = []

    try:
        # Lire toutes les feuilles du fichier Excel
        excel_file = pd.ExcelFile(str(chemin))

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)

            # Convertir le DataFrame en texte formaté
            if not df.empty:
                texte = f"# Feuille: {sheet_name}\n\n"
                texte += df.to_string(index=False)

                pages.append(ParsedPage(
                    texte=texte.strip(),
                    source=f"{chemin.name} (Feuille: {sheet_name})",
                    page=len(pages) + 1,
                ))
    except Exception as e:
        print(f"  Impossible de lire {chemin.name} : {e}")
        return []

    return pages
