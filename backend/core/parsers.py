"""
core/parsers.py — Parsers multi-format : PDF, DOCX, TXT/MD, CSV, Excel.

Chaque parser retourne une liste de ParsedPage : {"texte": str, "source": str, "page": int}

Docling est utilisé en priorité pour PDF et DOCX (meilleure extraction de tableaux,
compréhension du layout). Variables d'environnement :
    USE_DOCLING         : active Docling (défaut: true)
    DOCLING_OCR         : active l'OCR pour PDF scannés (défaut: false — lourd)
    DOCLING_TABLE_MODE  : fast ou accurate (défaut: fast)
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Configuration Docling ───────────────────────────────────────────────────
USE_DOCLING = os.environ.get("USE_DOCLING", "true").lower() == "true"
DOCLING_OCR = os.environ.get("DOCLING_OCR", "false").lower() == "true"
DOCLING_TABLE_MODE = os.environ.get("DOCLING_TABLE_MODE", "fast")  # fast | accurate

# Singleton lazy — initialisé au premier appel
_docling_converter = None


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


# ── Docling ─────────────────────────────────────────────────────────────────

def _get_docling_converter():
    """
    Initialisation lazy du converter Docling.
    Les modèles sont téléchargés au premier appel (~600MB sans OCR, ~1.5GB avec).
    """
    global _docling_converter
    if _docling_converter is not None:
        return _docling_converter

    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
        WordFormatOption,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    from docling.pipeline.simple_pipeline import SimplePipeline

    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = DOCLING_OCR
    pdf_options.do_table_structure = True
    pdf_options.table_structure_options = TableStructureOptions(
        mode=TableFormerMode.FAST if DOCLING_TABLE_MODE == "fast" else TableFormerMode.ACCURATE,
        do_cell_matching=True,
    )

    # XLSX : SimplePipeline si supporté par la version installée
    allowed_formats = [InputFormat.PDF, InputFormat.DOCX]
    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
    }
    try:
        if hasattr(InputFormat, "XLSX"):
            allowed_formats.append(InputFormat.XLSX)
            logger.info("Docling : support XLSX activé")
    except Exception:
        pass

    _docling_converter = DocumentConverter(
        allowed_formats=allowed_formats,
        format_options=format_options,
    )

    mode_str = f"OCR={'on' if DOCLING_OCR else 'off'}, table={DOCLING_TABLE_MODE}"
    logger.info(f"Docling initialisé ({mode_str})")
    return _docling_converter


def _docling_result_to_pages(result, chemin: Path) -> list[ParsedPage]:
    """
    Convertit un résultat Docling en liste de ParsedPage groupée par numéro de page.

    Utilise la provenance (prov) de chaque item pour récupérer le numéro de page.
    Les tableaux sont exportés en Markdown pour préserver leur structure.
    """
    from collections import defaultdict
    from docling_core.types.doc import TableItem, TextItem

    doc = result.document
    pages_content: dict[int, list[str]] = defaultdict(list)

    for item, _level in doc.iterate_items():
        text = None

        if isinstance(item, TableItem):
            # Tableau → Markdown avec headers (meilleur pour le LLM)
            try:
                df = item.export_to_dataframe(doc=doc)
                text = df.to_markdown(index=False)
            except Exception:
                try:
                    text = item.export_to_html(doc=doc)
                except Exception:
                    pass

        elif isinstance(item, TextItem):
            text = item.text

        if not text or not text.strip():
            continue

        # Récupérer le numéro de page depuis la provenance
        page_no = 1
        if hasattr(item, "prov") and item.prov:
            page_no = item.prov[0].page_no

        pages_content[page_no].append(text.strip())

    if not pages_content:
        # Fallback : export markdown complet
        markdown = doc.export_to_markdown()
        if markdown.strip():
            return [ParsedPage(texte=markdown.strip(), source=chemin.name, page=1)]
        return []

    return [
        ParsedPage(
            texte="\n\n".join(texts),
            source=chemin.name,
            page=page_no,
        )
        for page_no, texts in sorted(pages_content.items())
        if texts and any(t.strip() for t in texts)
    ]


# ── Parsers spécifiques ─────────────────────────────────────────────────────

def _parser_pdf(chemin: Path) -> list[ParsedPage]:
    """
    Extraction PDF.
    Priorité : Docling (layout AI + extraction tableaux)
    Fallback  : PyMuPDF4LLM
    """
    if USE_DOCLING:
        try:
            logger.info(f"Docling PDF : {chemin.name}")
            converter = _get_docling_converter()
            result = converter.convert(str(chemin))
            pages = _docling_result_to_pages(result, chemin)
            if pages:
                logger.info(f"Docling OK : {len(pages)} pages extraites")
                return pages
            logger.warning(f"Docling : aucun contenu extrait pour {chemin.name}, fallback PyMuPDF")
        except Exception as e:
            logger.warning(f"Docling échoué pour {chemin.name} : {e} — fallback PyMuPDF4LLM")

    return _parser_pdf_pymupdf(chemin)


def _parser_pdf_pymupdf(chemin: Path) -> list[ParsedPage]:
    """Fallback PDF via PyMuPDF4LLM."""
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
        num_page = num_page + 1 if isinstance(num_page, int) else 1

        if texte.strip():
            pages.append(ParsedPage(
                texte=texte.strip(),
                source=chemin.name,
                page=num_page,
            ))

    return pages


def _parser_docx(chemin: Path) -> list[ParsedPage]:
    """
    Extraction DOCX.
    Priorité : Docling (extrait les tableaux — python-docx les ignore)
    Fallback  : python-docx (paragraphes uniquement)
    """
    if USE_DOCLING:
        try:
            logger.info(f"Docling DOCX : {chemin.name}")
            converter = _get_docling_converter()
            result = converter.convert(str(chemin))
            pages = _docling_result_to_pages(result, chemin)
            if pages:
                logger.info(f"Docling OK : {len(pages)} pages extraites")
                return pages
            logger.warning(f"Docling : aucun contenu extrait pour {chemin.name}, fallback python-docx")
        except Exception as e:
            logger.warning(f"Docling échoué pour {chemin.name} : {e} — fallback python-docx")

    return _parser_docx_legacy(chemin)


def _parser_docx_legacy(chemin: Path) -> list[ParsedPage]:
    """Fallback DOCX via python-docx (paragraphes uniquement, sans tableaux)."""
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
    """
    Extraction Excel (.xlsx/.xls).
    Priorité : Docling (si InputFormat.XLSX disponible) — meilleure structure.
    Fallback  : pandas avec groupement par blocs de 30 lignes + markdown.
    """
    # Tentative Docling (ne supporte que .xlsx, pas .xls)
    if USE_DOCLING and chemin.suffix.lower() == ".xlsx":
        try:
            from docling.datamodel.base_models import InputFormat
            if hasattr(InputFormat, "XLSX"):
                converter = _get_docling_converter()
                result = converter.convert(str(chemin))
                pages = _docling_result_to_pages(result, chemin)
                if pages:
                    logger.info(f"Docling XLSX OK : {len(pages)} pages — {chemin.name}")
                    return pages
                logger.warning(f"Docling XLSX : aucun contenu — fallback pandas ({chemin.name})")
        except Exception as e:
            logger.warning(f"Docling XLSX échoué ({e}) — fallback pandas ({chemin.name})")

    return _parser_excel_pandas(chemin)


def _parser_excel_pandas(chemin: Path) -> list[ParsedPage]:
    """
    Fallback Excel via pandas.
    Groupe les lignes par blocs de 30 (headers répétés) pour éviter
    les micro-chunks (une ligne = un chunk) avec le semantic chunker.
    """
    import pandas as pd

    ROWS_PER_BLOCK = 30
    pages = []

    try:
        excel_file = pd.ExcelFile(str(chemin))

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            if df.empty:
                continue

            # Diviser en blocs de ROWS_PER_BLOCK lignes avec headers répétés
            for i in range(0, len(df), ROWS_PER_BLOCK):
                bloc = df.iloc[i: i + ROWS_PER_BLOCK]
                texte = f"# Feuille: {sheet_name} (lignes {i + 1}-{i + len(bloc)})\n\n"
                texte += bloc.to_markdown(index=False)
                pages.append(ParsedPage(
                    texte=texte.strip(),
                    source=f"{chemin.name} (Feuille: {sheet_name})",
                    page=len(pages) + 1,
                ))

    except Exception as e:
        logger.warning(f"Impossible de lire {chemin.name} : {e}")
        return []

    return pages
