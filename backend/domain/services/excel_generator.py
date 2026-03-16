"""
Excel devis generator — v3.

Template (templateDevis.xlsx) :
  Row 1  : Header
  Row 2  : Marge % col J (fond jaune, $J$2 absolu — pré-existant dans le template, modifiable par LLM)
  Row 3+ : Séparateurs catégorie + SubTotal

Trois niveaux outline :
  "1"    → seuls SubTotal catégorie
  "2"/[+]→ + TOTAL par poste (outline_level=1)
  "3"/[+]→ + composants (outline_level=2)

Rendu ligne composant :
  • Texte col D (nom_poste)   : gras Arial 9, noir
  • Texte col E (élément)     : italique Arial 9, gris (#808080), aligné droite
  • Texte col F (fournisseur) : italique Arial 9, gris (#808080), aligné droite
  • Prix I/J/X                : normal Arial 9, centré, format €
  • Bordures horizontales fines grises (#808080) entre chaque ligne composant (toutes colonnes)
  • Contour external medium noir autour du bloc poste (pas de séparateurs verticaux)
  • Bordure droite medium sur col D → délimite nom_poste même composants déroulés

Ligne TOTAL : normal Arial 9, blanc, prix centré format €.

Coefficient = pourcentage. Ex: 10 → 500 € devient 550 €. Défaut 0.
"""

import io
import logging
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Border, Color, Font, PatternFill, Side

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path("/app/documents/excelDoc/templateDevis.xlsx")
SHEET_NAME = "détail prix"

CATEGORY_ORDER = ["mechanical", "electrical", "process", "others"]

SUBTOTAL_MARKERS = {
    "mechanical": "Mechanical SubTotal",
    "electrical": "Electrical & Automation SubTotal",
    "process":    "Process Subtotal",
    "others":     "Others Subtotal",
}

AGGREGATE_COLS = list("IJKLMNOPQRSTUVWX")
TOTAL_COLS     = list("IJKLMNOPQRSTUVWX")

_MAX_COL       = 24
_COEF_CELL_REF = "$J$2"
_COL_NOM_POSTE = 4   # col D — reçoit une bordure droite medium


# ── Mapping catégorie ─────────────────────────────────────────────────────────────

_ENSEMBLE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("electrical", ["électr", "electr", "autom", "e/a", "e-a"]),
    ("process",    ["procéd", "proced", "procede", "process"]),
    ("others",     ["autre", "other"]),
]


def _map_category(ensemble: str | None) -> str:
    if not ensemble:
        return "mechanical"
    v = ensemble.lower().strip()
    for cat, kws in _ENSEMBLE_KEYWORDS:
        if any(kw in v for kw in kws):
            return cat
    return "mechanical"


# ── Constantes de style ───────────────────────────────────────────────────────────

_GREY_HEX    = "808080"          # gris — même couleur pour les traits et le texte E/F
_SIDE_MEDIUM = Side(style="medium")
_SIDE_NONE   = Side(style=None)
_SIDE_GREY   = Side(style="thin", color=_GREY_HEX)  # trait gris entre lignes composants

_FONT_NORMAL      = Font(name="Arial", size=9)
_FONT_BOLD        = Font(name="Arial", size=9, bold=True)
_FONT_ITALIC_GREY = Font(name="Arial", size=9, italic=True, color=_GREY_HEX)  # cols E et F

_FILL_COEF     = PatternFill(patternType="solid", fgColor="FFFFCC")  # fond jaune coeff
_FILL_ORANGE   = PatternFill(fill_type="solid",     fgColor="FFF9D2B7")  # orange Accentuation 2 +65% (ARGB)
_FILL_DIAGONAL = PatternFill(fill_type="lightUp", fgColor="FF000000", bgColor="FFFFFFFF")  # rayures fines diag.

_ALIGN_RIGHT  = Alignment(horizontal="right",  vertical="center", wrap_text=False)
_ALIGN_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=False)

_FORMAT_EURO = '#,##0.00 "€"'
_FORMAT_PCT  = '0%'


# ── Bordures ──────────────────────────────────────────────────────────────────────


def _apply_block_borders(ws, start_row: int, end_row: int) -> None:
    """
    Contour external medium noir autour du bloc poste.
    Pas de séparateurs horizontaux entre les lignes internes (pour ne pas
    couper le remplissage diagonal des lignes composants).
    Pas de séparateurs verticaux entre colonnes.
    """
    for row_n in range(start_row, end_row + 1):
        is_first = row_n == start_row
        is_last  = row_n == end_row
        for col_n in range(1, _MAX_COL + 1):
            top    = _SIDE_MEDIUM if is_first else _SIDE_NONE
            bottom = _SIDE_MEDIUM if is_last  else _SIDE_NONE
            left   = _SIDE_MEDIUM if col_n == 1        else _SIDE_NONE
            right  = _SIDE_MEDIUM if col_n == _MAX_COL else _SIDE_NONE
            ws.cell(row_n, col_n).border = Border(
                top=top, bottom=bottom, left=left, right=right,
            )


def _apply_nom_poste_right_border(ws, start_row: int, end_row: int) -> None:
    """
    Ajoute une bordure droite medium sur col D.
    Préserve les bordures top/bottom déjà posées par _apply_block_borders.
    """
    for row_n in range(start_row, end_row + 1):
        existing = ws.cell(row_n, _COL_NOM_POSTE).border
        ws.cell(row_n, _COL_NOM_POSTE).border = Border(
            top    = existing.top,
            bottom = existing.bottom,
            left   = existing.left,
            right  = _SIDE_MEDIUM,
        )



# ── Écrivains de lignes ───────────────────────────────────────────────────────────


def _write_element_row(
    ws,
    row_num: int,
    nom_poste: str,
    num_ensemble: str,
    ensemble: str,
    num_poste: str,
    elements_desc: str,
    fournisseur: str,
    fourniture,
) -> None:
    """
    Ligne composant (outline_level=2).

    Col D  : nom_poste (gras, noir, Arial 9)
    Col E  : élément (italique gris #808080, Arial 9, aligné droite)
    Col F  : fournisseur (italique gris #808080, Arial 9, aligné droite)
    Col I  : fourniture brut (centré, format €)
    Col J  : =I*(1+$J$2/100) — prix de vente (centré, format €)
    Col X  : =J — budget (centré, format €)
    Bordures : via _apply_block_borders (traits gris horizontaux + contour medium).
    """
    def _w(col: int, value, font=None, alignment=None, number_format=None):
        c = ws.cell(row_num, col, value=value)
        c.font = font if font is not None else _FONT_NORMAL
        if alignment:     c.alignment     = alignment
        if number_format: c.number_format = number_format

    # Fond blanc + rayures diagonales fines à partir de col K (11)
    for col_n in range(11, _MAX_COL + 1):
        ws.cell(row_num, col_n).fill = _FILL_DIAGONAL

    _w(1, num_ensemble or "")
    _w(2, ensemble or "")
    _w(3, num_poste or "")
    _w(4, nom_poste or "",     font=_FONT_BOLD)
    _w(5, elements_desc or "", font=_FONT_ITALIC_GREY, alignment=_ALIGN_RIGHT)
    _w(6, fournisseur or "",   font=_FONT_ITALIC_GREY, alignment=_ALIGN_RIGHT)

    price = None
    if fourniture is not None:
        try:
            price = float(str(fourniture).replace(",", ".").replace(" ", ""))
        except (ValueError, TypeError):
            price = None
    _w(9, price, alignment=_ALIGN_CENTER, number_format=_FORMAT_EURO)
    # Cols J, K–X : vides sur les lignes composants


def _write_total_row(
    ws,
    row_num: int,
    nom_poste: str,
    first_elem: int,
    last_elem: int,
    coef_ref: str | None = None,
    nbre_jours_etude:   int   = 0,
    mdo_etude:          float = 0.0,
    nbre_jours_atelier: int   = 0,
    mdo_atelier:        float = 0.0,
    nbre_jours_client:  int   = 0,
    mdo_client:         float = 0.0,
    coef_final:         float = 0.0,
) -> None:
    """
    Ligne TOTAL après le dernier composant (outline_level=1).
    Normal Arial 9, fond blanc. Prix centré format €.

    coef_ref   : référence absolue à la cellule coefficient principal (défaut = $J$2).
                 Pour la section OPTIONS, passer la cellule coeff options (ex: "$J$21").
    coef_final : taux de marge final (ex: 0.08 pour 8%). Écrit en col S (format %).
    """
    def _w(col: int, value, font=None, alignment=None, number_format=None):
        c = ws.cell(row_num, col, value=value)
        c.font = font if font is not None else _FONT_NORMAL
        if alignment:     c.alignment     = alignment
        if number_format: c.number_format = number_format

    effective_coef = coef_ref if coef_ref is not None else _COEF_CELL_REF

    # ── Cols I–J : fourniture brute + markup ─────────────────────────────────
    _w(4, nom_poste or "", font=_FONT_BOLD)
    _w(5, "TOTAL")
    _w(9,  f"=_xlfn.AGGREGATE(9,2,I{first_elem}:I{last_elem})",
       alignment=_ALIGN_CENTER, number_format=_FORMAT_EURO)
    _w(10, f"=I{row_num}*{effective_coef}",
       alignment=_ALIGN_CENTER, number_format=_FORMAT_EURO)

    # ── K(11) nbre jours étude — fond orange ─────────────────────────────────
    ws.cell(row_num, 11).fill = _FILL_ORANGE
    _w(11, nbre_jours_etude, alignment=_ALIGN_CENTER)

    # ── L(12) mdo étude = nbre_jours × taux journalier $L$2 ──────────────────
    _w(12, f"=K{row_num}*$L$2", alignment=_ALIGN_CENTER, number_format=_FORMAT_EURO)

    # ── M(13) nbre jours atelier — fond orange ───────────────────────────────
    ws.cell(row_num, 13).fill = _FILL_ORANGE
    _w(13, nbre_jours_atelier, alignment=_ALIGN_CENTER)

    # ── N(14) mdo atelier = nbre_jours × taux journalier $N$2 ────────────────
    _w(14, f"=M{row_num}*$N$2", alignment=_ALIGN_CENTER, number_format=_FORMAT_EURO)

    # ── O(15) nbre jours client — fond orange ────────────────────────────────
    ws.cell(row_num, 15).fill = _FILL_ORANGE
    _w(15, nbre_jours_client, alignment=_ALIGN_CENTER)

    # ── P(16) mdo client = nbre_jours × taux journalier $P$2 ─────────────────
    _w(16, f"=O{row_num}*$P$2", alignment=_ALIGN_CENTER, number_format=_FORMAT_EURO)

    # ── Q(17) total mdo ──────────────────────────────────────────────────────
    _w(17, f"=L{row_num}+N{row_num}+P{row_num}",
       alignment=_ALIGN_CENTER, number_format=_FORMAT_EURO)

    # ── R(18) total appros = fourniture + markup ──────────────────────────────
    _w(18, f"=I{row_num}+J{row_num}",
       alignment=_ALIGN_CENTER, number_format=_FORMAT_EURO)

    # ── S(19) coef final (décimal : 0.08 = 8%) ───────────────────────────────
    _w(19, coef_final, alignment=_ALIGN_CENTER, number_format=_FORMAT_PCT)

    # ── T(20) mdo avec coef ──────────────────────────────────────────────────
    _w(20, f"=Q{row_num}*(1+S{row_num})",
       alignment=_ALIGN_CENTER, number_format=_FORMAT_EURO)

    # ── U(21) total VLM4NC avec coef — réservé (non implémenté) ─────────────

    # ── V(22) total appros avec coef ─────────────────────────────────────────
    _w(22, f"=R{row_num}*(1+S{row_num})",
       alignment=_ALIGN_CENTER, number_format=_FORMAT_EURO)

    # ── W(23) total appros et mdo ─────────────────────────────────────────────
    _w(23, f"=T{row_num}+V{row_num}",
       alignment=_ALIGN_CENTER, number_format=_FORMAT_EURO)

    # ── X(24) BUDGET = ARRONDI(W, -2) ────────────────────────────────────────
    _w(24, f"=ROUND(W{row_num},-2)",
       alignment=_ALIGN_CENTER, number_format=_FORMAT_EURO)


# ── Générateur principal ──────────────────────────────────────────────────────────


def _resolve_postes(items: list[dict], catalog_adapter) -> list[dict]:
    """
    Transforme une liste d'items (panier ou options) en liste de postes avec leurs éléments.
    Retourne : [{ nom_poste, num_ensemble, ensemble, num_poste, rows: [...] }, ...]
    """
    postes = []
    for item in items:
        nom_poste   = item.get("nom_poste", "").strip()
        nom_affaire = item.get("nom_affaire") or None
        if not nom_poste:
            continue

        elements = catalog_adapter.get_elements_for_poste(nom_poste, nom_affaire=nom_affaire)
        if not elements and nom_affaire:
            elements = catalog_adapter.get_elements_for_poste(nom_poste)

        mdo_fields = {
            "nbre_jours_etude":   int(item.get("nbre_jours_etude",   0) or 0),
            "mdo_etude":          float(item.get("mdo_etude",          0.0) or 0.0),
            "nbre_jours_atelier": int(item.get("nbre_jours_atelier", 0) or 0),
            "mdo_atelier":        float(item.get("mdo_atelier",        0.0) or 0.0),
            "nbre_jours_client":  int(item.get("nbre_jours_client",  0) or 0),
            "mdo_client":         float(item.get("mdo_client",         0.0) or 0.0),
            "coef_final":         float(item.get("coef_final",         0.0) or 0.0),
        }

        if elements:
            first = elements[0]
            postes.append({
                "nom_poste":    nom_poste,
                "num_ensemble": str(first.get("num_ensemble", "") or ""),
                "ensemble":     str(first.get("ensemble", "") or ""),
                "num_poste":    str(first.get("num_poste", "") or ""),
                "rows": [
                    {
                        "elements":    str(e.get("elements", "") or ""),
                        "fournisseur": str(e.get("fournisseur", "") or ""),
                        "fourniture":  e.get("fourniture"),
                    }
                    for e in elements
                ],
                **mdo_fields,
            })
        else:
            postes.append({
                "nom_poste":    nom_poste,
                "num_ensemble": "",
                "ensemble":     str(item.get("ensemble", "") or ""),
                "num_poste":    "",
                "rows": [{
                    "elements":    "",
                    "fournisseur": str(item.get("fournisseur", "") or ""),
                    "fourniture":  item.get("fourniture"),
                }],
                **mdo_fields,
            })
    return postes


def _insert_postes(ws, postes: list[dict], insert_at: int, coef_ref: str | None = None) -> None:
    """
    Insère une liste de postes (outline_level=1/2) à partir de la ligne `insert_at`.
    Utilisé pour le panier principal (par catégorie) et pour la section OPTIONS.

    coef_ref : référence absolue à la cellule coefficient pour la formule J de chaque TOTAL.
               Défaut = $J$2 (panier principal). Pour OPTIONS, passer la cellule coeff options.
    """
    total_rows = sum(len(p["rows"]) + 1 for p in postes)
    if total_rows == 0:
        return

    ws.insert_rows(insert_at, total_rows)
    insert_row = insert_at

    for poste in postes:
        block_start    = insert_row
        first_elem_row = insert_row

        for elem in poste["rows"]:
            _write_element_row(
                ws            = ws,
                row_num       = insert_row,
                nom_poste     = poste["nom_poste"],
                num_ensemble  = poste["num_ensemble"],
                ensemble      = poste["ensemble"],
                num_poste     = poste["num_poste"],
                elements_desc = elem["elements"],
                fournisseur   = elem["fournisseur"],
                fourniture    = elem["fourniture"],
            )
            ws.row_dimensions[insert_row].outline_level = 2
            ws.row_dimensions[insert_row].hidden        = True
            insert_row += 1

        last_elem_row = insert_row - 1

        _write_total_row(
            ws                 = ws,
            row_num            = insert_row,
            nom_poste          = poste["nom_poste"],
            first_elem         = first_elem_row,
            last_elem          = last_elem_row,
            coef_ref           = coef_ref,
            nbre_jours_etude   = poste.get("nbre_jours_etude",   0),
            mdo_etude          = poste.get("mdo_etude",          0.0),
            nbre_jours_atelier = poste.get("nbre_jours_atelier", 0),
            mdo_atelier        = poste.get("mdo_atelier",        0.0),
            nbre_jours_client  = poste.get("nbre_jours_client",  0),
            mdo_client         = poste.get("mdo_client",         0.0),
            coef_final         = poste.get("coef_final",         0.0),
        )
        ws.row_dimensions[insert_row].outline_level = 1
        ws.row_dimensions[insert_row].hidden        = True
        insert_row += 1

        _apply_block_borders(ws, block_start, insert_row - 1)
        _apply_nom_poste_right_border(ws, block_start, insert_row - 1)


def generate_excel_devis(
    panier: list[dict],
    catalog_adapter,
    coefficient: float = 0.0,
    options: list[dict] | None = None,
) -> bytes:
    """
    Génère un devis Excel rempli à partir des items du panier.

    Args:
        panier:           items standards — insérés dans les sections catégorie (Mécanique…)
        catalog_adapter:  instance CatalogAdapter
        coefficient:      marge en % (défaut 0 = aucune marge).
                          Ex : 10 → prix de vente = fourniture × 1.10
                          Le LLM fixe cette valeur via le tool calling de devis_service.
        options:          items optionnels — insérés dans la section OPTIONS du template.
                          Même format que `panier`. Le client pourra accepter ou refuser
                          ces postes indépendamment du devis principal.

    Returns:
        bytes du fichier .xlsx généré
    """
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template Excel introuvable : {TEMPLATE_PATH}")

    wb = openpyxl.load_workbook(TEMPLATE_PATH)
    ws = wb[SHEET_NAME]

    try:
        from openpyxl.worksheet.properties import OutlinePr, WorksheetProperties
        if ws.sheet_properties is None:
            ws.sheet_properties = WorksheetProperties()
        ws.sheet_properties.outlinePr = OutlinePr(summaryBelow=True, summaryRight=False)
    except Exception:
        pass

    # La ligne Marge % (row 2, col J = $J$2) est pré-existante dans le template.
    # Si `coefficient` est fourni, on écrase la valeur actuelle de la cellule.
    if coefficient != 0.0:
        ws.cell(2, 10).value = coefficient

    # ── Étape 2 : Grouper les items par catégorie → par poste ────────────────────
    sections: dict[str, list[dict]] = {cat: [] for cat in CATEGORY_ORDER}

    for item in panier:
        nom_poste   = item.get("nom_poste", "").strip()
        nom_affaire = item.get("nom_affaire") or None
        if not nom_poste:
            continue

        elements = catalog_adapter.get_elements_for_poste(nom_poste, nom_affaire=nom_affaire)
        if not elements and nom_affaire:
            logger.warning(
                "No elements for nom_poste=%r affaire=%r — retrying without affaire filter",
                nom_poste, nom_affaire,
            )
            elements = catalog_adapter.get_elements_for_poste(nom_poste)

        mdo_fields = {
            "nbre_jours_etude":   int(item.get("nbre_jours_etude",   0) or 0),
            "mdo_etude":          float(item.get("mdo_etude",          0.0) or 0.0),
            "nbre_jours_atelier": int(item.get("nbre_jours_atelier", 0) or 0),
            "mdo_atelier":        float(item.get("mdo_atelier",        0.0) or 0.0),
            "nbre_jours_client":  int(item.get("nbre_jours_client",  0) or 0),
            "mdo_client":         float(item.get("mdo_client",         0.0) or 0.0),
            "coef_final":         float(item.get("coef_final",         0.0) or 0.0),
        }

        if elements:
            first = elements[0]
            cat   = _map_category(first.get("ensemble") or item.get("ensemble"))
            sections[cat].append({
                "nom_poste":     nom_poste,
                "num_ensemble":  str(first.get("num_ensemble", "") or ""),
                "ensemble":      str(first.get("ensemble", "") or ""),
                "num_poste":     str(first.get("num_poste", "") or ""),
                "rows": [
                    {
                        "elements":    str(e.get("elements", "") or ""),
                        "fournisseur": str(e.get("fournisseur", "") or ""),
                        "fourniture":  e.get("fourniture"),
                    }
                    for e in elements
                ],
                **mdo_fields,
            })
        else:
            cat = _map_category(item.get("ensemble"))
            sections[cat].append({
                "nom_poste":    nom_poste,
                "num_ensemble": "",
                "ensemble":     str(item.get("ensemble", "") or ""),
                "num_poste":    "",
                "rows": [
                    {
                        "elements":    "",
                        "fournisseur": str(item.get("fournisseur", "") or ""),
                        "fourniture":  item.get("fourniture"),
                    }
                ],
                **mdo_fields,
            })

    logger.info("Devis sections: %s", {cat: len(p) for cat, p in sections.items()})

    # ── Étape 3 : Localiser SubTotal et TOTAL GÉNÉRAL ─────────────────────────────
    original_subtotal_rows: dict[str, int] = {}
    original_total_row: int | None = None

    for row in ws.iter_rows():
        rn    = row[0].row
        d_val = str(ws.cell(rn, 4).value or "")
        e_val = str(ws.cell(rn, 5).value or "")

        for cat, marker in SUBTOTAL_MARKERS.items():
            if marker.lower() in d_val.lower():
                original_subtotal_rows[cat] = rn

        if "TOTAL" in e_val and "GENERAL" in e_val:
            original_total_row = rn

    logger.info("SubTotal rows: %s, TOTAL GÉNÉRAL: %s", original_subtotal_rows, original_total_row)

    # ── Étape 4 : Insérer les lignes ──────────────────────────────────────────────
    offset = 0
    final_subtotal_rows: dict[str, int]              = {}
    final_data_ranges:   dict[str, tuple[int, int] | None] = {}

    for cat in CATEGORY_ORDER:
        orig_sub = original_subtotal_rows.get(cat)
        if orig_sub is None:
            logger.warning("SubTotal catégorie %r introuvable", cat)
            continue

        current_sub = orig_sub + offset
        postes      = sections[cat]
        total_rows  = sum(len(p["rows"]) + 1 for p in postes)

        if total_rows > 0:
            ws.insert_rows(current_sub, total_rows)
            insert_row = current_sub

            for poste in postes:
                block_start    = insert_row
                first_elem_row = insert_row

                for elem in poste["rows"]:
                    _write_element_row(
                        ws            = ws,
                        row_num       = insert_row,
                        nom_poste     = poste["nom_poste"],
                        num_ensemble  = poste["num_ensemble"],
                        ensemble      = poste["ensemble"],
                        num_poste     = poste["num_poste"],
                        elements_desc = elem["elements"],
                        fournisseur   = elem["fournisseur"],
                        fourniture    = elem["fourniture"],
                    )
                    ws.row_dimensions[insert_row].outline_level = 2
                    ws.row_dimensions[insert_row].hidden        = True
                    insert_row += 1

                last_elem_row = insert_row - 1

                _write_total_row(
                    ws                 = ws,
                    row_num            = insert_row,
                    nom_poste          = poste["nom_poste"],
                    first_elem         = first_elem_row,
                    last_elem          = last_elem_row,
                    nbre_jours_etude   = poste.get("nbre_jours_etude",   0),
                    mdo_etude          = poste.get("mdo_etude",          0.0),
                    nbre_jours_atelier = poste.get("nbre_jours_atelier", 0),
                    mdo_atelier        = poste.get("mdo_atelier",        0.0),
                    nbre_jours_client  = poste.get("nbre_jours_client",  0),
                    mdo_client         = poste.get("mdo_client",         0.0),
                    coef_final         = poste.get("coef_final",         0.0),
                )
                ws.row_dimensions[insert_row].outline_level = 1
                ws.row_dimensions[insert_row].hidden        = True
                insert_row += 1

                # 1. Contour medium + traits gris horizontaux entre lignes
                _apply_block_borders(ws, block_start, insert_row - 1)
                # 2. Bordure droite medium sur col D (préserve top/bottom)
                _apply_nom_poste_right_border(ws, block_start, insert_row - 1)

            offset               += total_rows
            final_data_ranges[cat] = (current_sub, current_sub + total_rows - 1)
            current_sub            = current_sub + total_rows
        else:
            final_data_ranges[cat] = None

        final_subtotal_rows[cat] = current_sub

    # ── Étape 5 : AGGREGATE SubTotal ─────────────────────────────────────────────
    for cat in CATEGORY_ORDER:
        sub_row = final_subtotal_rows.get(cat)
        if sub_row is None:
            continue
        dr = final_data_ranges.get(cat)
        for col_letter in AGGREGATE_COLS:
            if dr:
                start_row, end_row = dr
                formula = f"=_xlfn.AGGREGATE(9,2,{col_letter}{start_row}:{col_letter}{end_row})"
            else:
                formula = "=0"
            ws[f"{col_letter}{sub_row}"] = formula

    # ── Étape 6 : TOTAL GÉNÉRAL ───────────────────────────────────────────────────
    if original_total_row is not None:
        actual_total_row = original_total_row + offset
        sub_rows = [
            final_subtotal_rows[cat]
            for cat in CATEGORY_ORDER
            if cat in final_subtotal_rows
        ]
        if len(sub_rows) >= 2:
            for col_letter in TOTAL_COLS:
                refs = "+".join(f"{col_letter}{r}" for r in sub_rows)
                ws[f"{col_letter}{actual_total_row}"] = f"={refs}"

    # ── Étape 7 : Insérer les postes OPTIONS ─────────────────────────────────────
    if options:
        options_postes = _resolve_postes(options, catalog_adapter)
        if options_postes:
            # Structure du template (relative au header) :
            #   header     : en-tête OPTIONS (E="OPTIONS")
            #   header + 1 : ligne coeff options (ne pas toucher)
            #   header + 2 : SubTotal Option (A="O") ← décalé après insertion
            #   header + 3 : TOTAL OPTIONS           ← décalé après insertion
            options_header_row = None
            for row in ws.iter_rows():
                rn    = row[0].row
                e_val = str(ws.cell(rn, 5).value or "")
                # "OPTIONS" seul dans col E, pas "TOTAL OPTIONS"
                if "OPTIONS" in e_val.upper() and "TOTAL" not in e_val.upper():
                    options_header_row = rn
                    break

            if options_header_row is None:
                logger.warning("Ligne OPTIONS introuvable dans le template — options ignorées")
            else:
                insert_at          = options_header_row + 2
                total_options_rows = sum(len(p["rows"]) + 1 for p in options_postes)

                # Positions APRÈS insertion
                actual_subtotal      = insert_at + total_options_rows      # ex-header+2
                actual_total_options = insert_at + total_options_rows + 1  # ex-header+3

                # Cellule coeff spécifique aux options (ligne juste après le header)
                options_coef_ref = f"$J${options_header_row + 1}"

                _insert_postes(ws, options_postes, insert_at, coef_ref=options_coef_ref)

                # Mettre à jour AGGREGATE du SubTotal options
                data_start = insert_at
                data_end   = insert_at + total_options_rows - 1
                for col_letter in AGGREGATE_COLS:
                    ws[f"{col_letter}{actual_subtotal}"] = (
                        f"=_xlfn.AGGREGATE(9,2,{col_letter}{data_start}:{col_letter}{data_end})"
                    )

                # Mettre à jour TOTAL OPTIONS → référence directe au SubTotal
                for col_letter in AGGREGATE_COLS:
                    ws[f"{col_letter}{actual_total_options}"] = f"={col_letter}{actual_subtotal}"

                logger.info(
                    "Options insérées : %d postes R%d:R%d | SubTotal R%d | TOTAL OPTIONS R%d",
                    len(options_postes), data_start, data_end,
                    actual_subtotal, actual_total_options,
                )

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    logger.info("Excel devis généré : %d items, %d lignes insérées", len(panier), offset)
    return output.read()
