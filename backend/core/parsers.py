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
    parser: str = "docling"  # "docling" | "pymupdf" | "docx" | "text" | "csv" | "excel"


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


def _df_to_prose(df) -> str:
    """
    Convertit un DataFrame en phrases lisibles pour améliorer l'embedding.

    Exemple :
      | Fonction    | Basculeur | Plateau |
      | Charge maxi | 4.5T      | 3T      |
    →  "Charge maxi : Basculeur = 4.5T, Plateau = 3T."
    """
    lines = []
    col_names = [str(c).strip() for c in df.columns]

    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row]
        # Ignorer les lignes entièrement vides ou NaN
        non_empty = [v for v in vals if v and v.lower() not in ("nan", "none", "")]
        if not non_empty:
            continue

        first = vals[0] if vals[0].lower() not in ("nan", "none", "") else ""
        parts = []
        for col, val in zip(col_names[1:], vals[1:]):
            if val and val.lower() not in ("nan", "none", ""):
                parts.append(f"{col} = {val}")

        if parts:
            prefix = f"{first} : " if first else ""
            lines.append(f"{prefix}{', '.join(parts)}.")
        elif first:
            lines.append(first + ".")

    return "\n".join(lines)


def _docling_result_to_pages(result, chemin: Path) -> list[ParsedPage]:
    """
    Convertit un résultat Docling en liste de ParsedPage groupée par numéro de page.

    AMÉLIORATION : Préserve les métadonnées de structure (labels, niveaux)
    pour une meilleure détection des titres et sections.
    """
    from collections import defaultdict
    from docling_core.types.doc import TableItem, TextItem, DocItemLabel, PictureItem

    doc = result.document
    pages_content: dict[int, list[str]] = defaultdict(list)
    last_section = ""  # titre de la dernière section rencontrée (parent immédiat)

    for item, level in doc.iterate_items():
        text = None
        metadata_prefix = ""

        if isinstance(item, TableItem):
            # Tableau → Markdown (LLM lit et reproduit le tableau)
            #           + prose  (embedding trouve le chunk de façon fiable)
            try:
                df = item.export_to_dataframe(doc=doc)
                markdown = df.to_markdown(index=False)
                prose = _df_to_prose(df)
                # Légende du tableau si disponible (ex: "Tableau 3 : Caractéristiques du vireur")
                caption = item.caption_text(doc).strip() if hasattr(item, "caption_text") else ""
                parts = []
                if caption:
                    parts.append(caption)
                parts.append(markdown)
                if prose:
                    parts.append(prose)
                text = "\n\n".join(parts)
                section_ref = f"[{last_section}] " if last_section else ""
                metadata_prefix = f"[TABLE-L{level}]{section_ref}"
            except Exception:
                try:
                    text = item.export_to_html(doc=doc)
                    section_ref = f"[{last_section}] " if last_section else ""
                    metadata_prefix = f"[TABLE-L{level}]{section_ref}"
                except Exception:
                    pass

        elif isinstance(item, PictureItem):
            # Image : pas d'analyse visuelle, juste un label [IMAGE] + légende Docling si dispo.
            # Évite les parents "vide" (titre seul) sur les pages constituées d'un titre + image.
            caption = ""
            try:
                caption = item.caption_text(doc).strip() if hasattr(item, "caption_text") else ""
            except Exception:
                pass
            section_ref = f"[{last_section}]" if last_section else ""
            text = f"[IMAGE]{section_ref}" + (f" — {caption}" if caption else "")
            metadata_prefix = f"[IMAGE-L{level}] "

        elif isinstance(item, TextItem):
            text = item.text

            # Enrichir avec les métadonnées de structure
            label = getattr(item, 'label', None)
            if label:
                if label in [DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER]:
                    # Mettre à jour le parent immédiat (tronqué à 40 chars)
                    last_section = text.strip()[:40]
                    tag = "TITRE" if label == DocItemLabel.TITLE else "SECTION"
                    metadata_prefix = f"[{tag}-L{level}] "
                else:
                    section_ref = f"[{last_section}]" if last_section else ""
                    metadata_prefix = f"[{label.value.upper()}-L{level}]{section_ref} "

        if not text or not text.strip():
            continue

        # Récupérer le numéro de page depuis la provenance
        page_no = 1
        if hasattr(item, "prov") and item.prov:
            page_no = item.prov[0].page_no

        # Ajouter le texte avec ses métadonnées
        enriched_text = f"{metadata_prefix}{text.strip()}"
        pages_content[page_no].append(enriched_text)

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
                parser="pymupdf",
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
    Filtre les feuilles cachées/très cachées (.xls via xlrd).
    """
    import pandas as pd

    ROWS_PER_BLOCK = 30
    pages = []

    # Détection visibilité feuilles pour .xls (xlrd expose hidden/vhidden)
    hidden_sheets: set[str] = set()
    if chemin.suffix.lower() == ".xls":
        try:
            import xlrd
            wb = xlrd.open_workbook(str(chemin))
            for i in range(wb.nsheets):
                vis = wb.sheet_visibility(i)  # 0=visible, 1=hidden, 2=très caché
                name = wb.sheet_names()[i]
                if vis != 0:
                    hidden_sheets.add(name)
            logger.info(
                f"Excel .xls {chemin.name} : {wb.nsheets} feuilles total, "
                f"{len(hidden_sheets)} cachées — {list(hidden_sheets)[:5]}"
            )
        except Exception as e:
            logger.debug(f"xlrd visibility check échoué ({e}) — toutes feuilles incluses")

    try:
        excel_file = pd.ExcelFile(str(chemin))
        all_sheets = excel_file.sheet_names
        visible_sheets = [s for s in all_sheets if s not in hidden_sheets]
        skipped = len(all_sheets) - len(visible_sheets)
        if skipped:
            logger.info(f"Excel {chemin.name} : {skipped} feuilles cachées ignorées sur {len(all_sheets)}")

        for sheet_name in visible_sheets:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            if df.empty:
                continue

            logger.debug(f"  Feuille '{sheet_name}' : {len(df)} lignes × {len(df.columns)} colonnes")

            # Diviser en blocs de ROWS_PER_BLOCK lignes avec headers répétés
            for i in range(0, len(df), ROWS_PER_BLOCK):
                bloc = df.iloc[i: i + ROWS_PER_BLOCK]
                header = f"# Feuille: {sheet_name} (lignes {i + 1}-{i + len(bloc)})\n\n"
                prose = _df_to_prose(bloc)
                texte = header + bloc.to_markdown(index=False)
                if prose:
                    texte += "\n\n" + prose
                pages.append(ParsedPage(
                    texte=texte.strip(),
                    source=f"{chemin.name} (Feuille: {sheet_name})",
                    page=len(pages) + 1,
                    parser="excel",
                ))

    except Exception as e:
        logger.warning(f"Impossible de lire {chemin.name} : {e}")
        return []

    logger.info(f"Excel {chemin.name} : {len(pages)} pages parsées")
    return pages
