"""
Excel devis generator.

Loads templateDevis.xlsx (which already has styles, headers, SubTotal rows and
TOTAL GÉNÉRAL formulas) and fills it with panier items from the catalog.

Template structure — blank template (rows shift when data is inserted):
  Row 1  : Header (Num_Ensemble | ENSEMBLE | Num_Poste | NOM POSTE | DETAIL DES PRIX | ...)
  Row 2  : Mechanical SubTotal   (AGGREGATE formulas, cols J–X)
  Row 3  : Grey separator — ELECTRIQUE - AUTOMATISME
  Row 4  : E-A SubTotal          (same)
  Row 5  : Grey separator — PROCEDE
  Row 6  : Process SubTotal      (same)
  Row 7  : Grey separator — AUTRES
  Row 8  : Others SubTotal       (same)
  Row 9  : TOTAL GÉNÉRAL         (=J2+J4+J6+J8, …)
  Rows 11-14 : OPTIONS section

Strategy:
  1. Detect SubTotal rows by scanning column D for known text markers.
  2. Group panier items by category (from "ensemble" field).
  3. For each category (Mechanical → E-A → Process → Others), insert data rows
     *before* the SubTotal row, tracking cumulative offset.
  4. Update AGGREGATE formulas in SubTotal rows to cover the inserted data rows.
  5. Update TOTAL GÉNÉRAL formulas to reference the new SubTotal row positions.
  6. Return the workbook as bytes.
"""

import io
import logging
from pathlib import Path

import openpyxl
from openpyxl.styles import Border, Side

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path("/app/documents/excelDoc/templateDevis.xlsx")
SHEET_NAME = "détail prix"

# Ordered list of category keys (process order matters for offset tracking)
CATEGORY_ORDER = ["mechanical", "electrical", "process", "others"]

# Text in column D that identifies each SubTotal row
SUBTOTAL_MARKERS = {
    "mechanical": "Mechanical SubTotal",
    "electrical": "Electrical & Automation SubTotal",
    "process": "Process Subtotal",
    "others": "Others Subtotal",
}

# Category A column code and B column label per category
CATEGORY_META = {
    "mechanical": ("M", "Mécanique"),
    "electrical": ("E/A", "Elec-Automatisme"),
    "process": ("P", "Procédé"),
    "others": ("A", "Autres"),
}

# Columns for which SubTotal AGGREGATE is updated (J through X = cols 10–24)
# We also update col I (Fourniture) even though the blank template skips it.
AGGREGATE_COLS = list("IJKLMNOPQRSTUVWX")

# Columns updated in TOTAL GÉNÉRAL (same set)
TOTAL_COLS = list("IJKLMNOPQRSTUVWX")


# ── Category mapping ────────────────────────────────────────────────────────────


_ENSEMBLE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("electrical", ["électr", "electr", "autom", "e/a", "e-a"]),
    ("process",    ["procéd", "proced", "procede", "process"]),
    ("others",     ["autre", "other"]),
]


def _map_category(ensemble: str | None) -> str:
    """Map an ensemble string to a category key (default: mechanical)."""
    if not ensemble:
        return "mechanical"
    v = ensemble.lower().strip()
    for cat, kws in _ENSEMBLE_KEYWORDS:
        if any(kw in v for kw in kws):
            return cat
    return "mechanical"


# ── Style helpers ───────────────────────────────────────────────────────────────


def _thin_border() -> Border:
    s = Side(style="thin")
    return Border(left=s, right=s, top=s, bottom=s)


# ── Data row writer ─────────────────────────────────────────────────────────────


def _write_data_row(
    ws,
    row_num: int,
    nom_poste: str,
    elements_desc: str,
    fournisseur: str,
    fourniture,
    num_ensemble_val: str,
    ensemble_val: str,
    num_poste_val: str,
) -> None:
    """Write a single data row into the worksheet at row_num."""
    border = _thin_border()

    def _write(col: int, value):
        c = ws.cell(row_num, col, value=value)
        c.border = border

    # A (1): Num_Ensemble — use catalog value or fallback to category code
    _write(1, num_ensemble_val or "")

    # B (2): ENSEMBLE — catalog ensemble value
    _write(2, ensemble_val or "")

    # C (3): Num_Poste
    _write(3, num_poste_val or "")

    # D (4): NOM POSTE
    _write(4, nom_poste or "")

    # E (5): DETAIL DES PRIX — elements description
    _write(5, elements_desc or "")

    # F (6): Fournisseur
    _write(6, fournisseur or "")

    # I (9): Fourniture (unit price — informational, not aggregated in template SubTotal)
    price = None
    if fourniture is not None:
        try:
            price = float(str(fourniture).replace(",", ".").replace(" ", ""))
        except (ValueError, TypeError):
            price = None
    _write(9, price)

    # X (24): BUDGET — same as fourniture for now (coef=1, no MDO)
    _write(24, price)


# ── Main generator ──────────────────────────────────────────────────────────────


def generate_excel_devis(panier: list[dict], catalog_adapter) -> bytes:
    """
    Generate a filled Excel devis from panier items.

    Args:
        panier: list of panier items (nom_poste, ensemble, fournisseur, fourniture, …)
        catalog_adapter: CatalogAdapter instance (for get_elements_for_poste)

    Returns:
        bytes of the generated .xlsx file
    """
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template Excel introuvable : {TEMPLATE_PATH}")

    wb = openpyxl.load_workbook(TEMPLATE_PATH)
    ws = wb[SHEET_NAME]

    # ── Step 1: Group panier by category, enrich with catalog elements ──────────
    sections: dict[str, list[dict]] = {cat: [] for cat in CATEGORY_ORDER}

    for item in panier:
        nom_poste   = item.get("nom_poste", "").strip()
        nom_affaire = item.get("nom_affaire") or None
        if not nom_poste:
            continue

        # Fetch catalog rows for this nom_poste, restricted to its affaire
        elements = catalog_adapter.get_elements_for_poste(nom_poste, nom_affaire=nom_affaire)

        # Fallback: if affaire filter returns nothing, try without affaire constraint
        if not elements and nom_affaire:
            logger.warning(
                "No elements found for nom_poste=%r affaire=%r — retrying without affaire filter",
                nom_poste, nom_affaire,
            )
            elements = catalog_adapter.get_elements_for_poste(nom_poste)

        if elements:
            for elem in elements:
                cat = _map_category(elem.get("ensemble") or item.get("ensemble"))
                sections[cat].append({
                    "nom_poste": nom_poste,
                    "elements": str(elem.get("elements", "") or ""),
                    "fournisseur": str(elem.get("fournisseur", "") or ""),
                    "fourniture": elem.get("fourniture"),
                    "ensemble": str(elem.get("ensemble", "") or ""),
                    "num_poste": str(elem.get("num_poste", "") or ""),
                    "num_ensemble": str(elem.get("num_ensemble", "") or ""),
                })
        else:
            # No catalog elements: use the panier item directly as a single row
            cat = _map_category(item.get("ensemble"))
            sections[cat].append({
                "nom_poste": nom_poste,
                "elements": "",
                "fournisseur": str(item.get("fournisseur", "") or ""),
                "fourniture": item.get("fourniture"),
                "ensemble": str(item.get("ensemble", "") or ""),
                "num_poste": "",
                "num_ensemble": "",
            })

    logger.info(
        "Devis sections: %s",
        {cat: len(rows) for cat, rows in sections.items()},
    )

    # ── Step 2: Locate SubTotal rows and TOTAL GÉNÉRAL row in template ──────────
    original_subtotal_rows: dict[str, int] = {}
    original_total_row: int | None = None

    for row in ws.iter_rows():
        rn = row[0].row
        d_val = str(ws.cell(rn, 4).value or "")
        e_val = str(ws.cell(rn, 5).value or "")

        for cat, marker in SUBTOTAL_MARKERS.items():
            if marker.lower() in d_val.lower():
                original_subtotal_rows[cat] = rn

        if "TOTAL" in e_val and "GENERAL" in e_val:
            original_total_row = rn

    logger.info("SubTotal rows: %s, TOTAL GÉNÉRAL: %s", original_subtotal_rows, original_total_row)

    # ── Step 3: Insert data rows (top → bottom, tracking cumulative offset) ─────
    offset = 0
    final_subtotal_rows: dict[str, int] = {}
    final_data_ranges: dict[str, tuple[int, int] | None] = {}

    for cat in CATEGORY_ORDER:
        orig_sub = original_subtotal_rows.get(cat)
        if orig_sub is None:
            logger.warning("SubTotal row for category %r not found in template", cat)
            continue

        current_sub = orig_sub + offset
        data_rows = sections[cat]
        N = len(data_rows)

        if N > 0:
            ws.insert_rows(current_sub, N)

            for i, row_data in enumerate(data_rows):
                _write_data_row(
                    ws=ws,
                    row_num=current_sub + i,
                    nom_poste=row_data["nom_poste"],
                    elements_desc=row_data["elements"],
                    fournisseur=row_data["fournisseur"],
                    fourniture=row_data["fourniture"],
                    num_ensemble_val=row_data["num_ensemble"],
                    ensemble_val=row_data["ensemble"],
                    num_poste_val=row_data["num_poste"],
                )

            offset += N
            final_data_ranges[cat] = (current_sub, current_sub + N - 1)
            current_sub = current_sub + N
        else:
            final_data_ranges[cat] = None

        final_subtotal_rows[cat] = current_sub

    # ── Step 4: Update AGGREGATE formulas in SubTotal rows ──────────────────────
    for cat in CATEGORY_ORDER:
        sub_row = final_subtotal_rows.get(cat)
        if sub_row is None:
            continue

        dr = final_data_ranges.get(cat)
        for col_letter in AGGREGATE_COLS:
            if dr:
                start_row, end_row = dr
                formula = (
                    f"=_xlfn.AGGREGATE(9,2,"
                    f"{col_letter}{start_row}:{col_letter}{end_row})"
                )
            else:
                formula = "=0"
            ws[f"{col_letter}{sub_row}"] = formula

    # ── Step 5: Update TOTAL GÉNÉRAL formulas ───────────────────────────────────
    if original_total_row is not None:
        actual_total_row = original_total_row + offset

        # Collect actual SubTotal rows in CATEGORY_ORDER
        sub_rows = [
            final_subtotal_rows[cat]
            for cat in CATEGORY_ORDER
            if cat in final_subtotal_rows
        ]

        if len(sub_rows) >= 2:
            for col_letter in TOTAL_COLS:
                refs = "+".join(f"{col_letter}{r}" for r in sub_rows)
                ws[f"{col_letter}{actual_total_row}"] = f"={refs}"

    # ── Save to bytes ────────────────────────────────────────────────────────────
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    logger.info("Excel devis generated (%d panier items, %d offset rows)", len(panier), offset)
    return output.read()
