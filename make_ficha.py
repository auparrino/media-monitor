"""
Cono Sur Media Monitor — 2-page project ficha.
Page 1: single-column flow (no overlaps)
Page 2: 3 screenshots + features + QR + DATA band
"""

import os, io
import qrcode
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

BASE  = os.path.dirname(os.path.abspath(__file__))
SS    = os.path.join(BASE, "output", "_dashboard_screenshot.png")
OUT   = os.path.join(BASE, "output", "ficha_conosur_monitor_v2.pdf")

NAVY      = HexColor("#003049")
CRIMSON   = HexColor("#C1121F")
CREAM     = HexColor("#FDF0D5")
CREAM_DRK = HexColor("#EDE0C0")
BLUE_GREY = HexColor("#669BBC")
WHITE     = HexColor("#FFFFFF")
DARK      = HexColor("#1A1A1A")
MID       = HexColor("#555555")
SHADOW    = HexColor("#C8BC9E")

W, H   = A4                # 595.28 × 841.89 pt
M      = 16 * mm
HDR_H  = 14 * mm
FTR_H  = 8.5 * mm
INNER  = W - 2 * M        # usable width


# ── Screenshot crops ──────────────────────────────────────────────────────────

def prepare_crops():
    """Crop the dashboard screenshot into 3 focused regions."""
    if not os.path.exists(SS):
        return None, None, None
    img = PILImage.open(SS)
    W2, H2 = img.size
    # Remove side cream margins: content starts ~x=260, ends ~x=2110
    cx1, cx2 = 260, 2110
    # Crop 1 — hero: letterhead + overview (top 38%)
    c1 = img.crop((cx1, 0,        cx2, int(H2 * 0.38)))
    # Crop 2 — key stories detail (25%–68%)
    c2 = img.crop((cx1, int(H2 * 0.25), cx2, int(H2 * 0.68)))
    # Crop 3 — also reported (58%–95%)
    c3 = img.crop((cx1, int(H2 * 0.58), cx2, int(H2 * 0.95)))
    paths = []
    for i, crop in enumerate([c1, c2, c3], 1):
        p = os.path.join(BASE, "output", f"_ss_crop{i}.png")
        crop.save(p)
        paths.append(p)
    return paths


def img_reader(path):
    if path and os.path.exists(path):
        return ImageReader(path)
    return None


def draw_img(c, path, x, y_top, max_w, max_h, shadow=True, radius=0):
    """Draw image scaled to fit. Returns actual height drawn (0 if no file)."""
    ir = img_reader(path)
    if ir is None:
        return 0
    iw, ih = ir.getSize()
    scale = min(max_w / iw, max_h / ih)
    dw, dh = iw * scale, ih * scale
    ix = x + (max_w - dw) / 2
    iy = y_top - dh
    if shadow:
        c.setFillColor(SHADOW)
        c.rect(ix + 2, iy - 2, dw, dh, fill=1, stroke=0)
    c.drawImage(path, ix, iy, dw, dh)
    c.setStrokeColor(HexColor("#BBBBBB"))
    c.setLineWidth(0.5)
    c.rect(ix, iy, dw, dh, fill=0, stroke=1)
    return dh


# ── Reusable chrome ───────────────────────────────────────────────────────────

def draw_header(c, left, right=""):
    c.setFillColor(NAVY)
    c.rect(0, H - HDR_H, W, HDR_H, fill=1, stroke=0)
    c.setFillColor(CREAM)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(M, H - HDR_H + 4.2 * mm, left)
    if right:
        c.setFillColor(BLUE_GREY)
        c.setFont("Helvetica", 7.5)
        c.drawRightString(W - M, H - HDR_H + 4.5 * mm, right)
    c.setStrokeColor(CRIMSON)
    c.setLineWidth(2.5)
    c.line(0, H - HDR_H - 0.7 * mm, W, H - HDR_H - 0.7 * mm)


def draw_footer(c, left, right):
    c.setFillColor(NAVY)
    c.rect(0, 0, W, FTR_H, fill=1, stroke=0)
    c.setFillColor(BLUE_GREY)
    c.setFont("Helvetica", 6.5)
    c.drawString(M, 2.5 * mm, left)
    c.drawRightString(W - M, 2.5 * mm, right)


def cream_page(c):
    c.setFillColor(CREAM)
    c.rect(0, 0, W, H, fill=1, stroke=0)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Overview (strict top-to-bottom flow)
# ══════════════════════════════════════════════════════════════════════════════

def page1(c, crop1):
    cream_page(c)
    draw_header(c, "CONO SUR  MEDIA MONITOR", "Proyecto personal · 2025–2026")
    draw_footer(c, "Cono Sur Media Monitor",
                "Argentina  ·  Uruguay  ·  Paraguay")

    # y tracks the current drawing position (top of next element)
    y = H - HDR_H - 5 * mm

    # ── Red tag + project line ────────────────────────────────────────────────
    TAG_W = 25 * mm
    c.setFillColor(CRIMSON)
    c.roundRect(M, y - 5 * mm, TAG_W, 5 * mm, 1.2 * mm, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 7.5)
    c.drawCentredString(M + TAG_W / 2, y - 3.6 * mm, "CONO SUR")
    c.setFillColor(MID)
    c.setFont("Helvetica", 8)
    c.drawString(M + TAG_W + 4 * mm, y - 3.6 * mm, "Proyecto personal  ·  2025–2026")
    y -= 8 * mm

    # ── Title ─────────────────────────────────────────────────────────────────
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 26)
    c.drawString(M, y, "MEDIA MONITOR")
    y -= 6 * mm
    c.setFillColor(BLUE_GREY)
    c.setFont("Helvetica", 9)
    c.drawString(M, y, "Briefing diario automatizado del Cono Sur")
    y -= 3 * mm

    # ── Crimson rule ──────────────────────────────────────────────────────────
    c.setStrokeColor(CRIMSON)
    c.setLineWidth(1.5)
    c.line(M, y, W - M, y)
    y -= 6 * mm

    # ── Stat cards ────────────────────────────────────────────────────────────
    CARD_W = 38 * mm
    CARD_H = 17 * mm
    GAP    = (INNER - 3 * CARD_W) / 2
    stats  = [("14", "diarios"), ("3", "países"), ("Diario", "actualización")]
    for i, (num, lbl) in enumerate(stats):
        cx = M + i * (CARD_W + GAP)
        c.setFillColor(NAVY)
        c.roundRect(cx, y - CARD_H, CARD_W, CARD_H, 2 * mm, fill=1, stroke=0)
        c.setFillColor(CREAM)
        c.setFont("Helvetica-Bold", 18)
        nw = c.stringWidth(num, "Helvetica-Bold", 18)
        c.drawString(cx + CARD_W / 2 - nw / 2, y - CARD_H + 6.5 * mm, num)
        c.setFillColor(BLUE_GREY)
        c.setFont("Helvetica", 7)
        lw = c.stringWidth(lbl, "Helvetica", 7)
        c.drawString(cx + CARD_W / 2 - lw / 2, y - CARD_H + 2 * mm, lbl)
    y -= CARD_H + 6 * mm

    # ── Description ───────────────────────────────────────────────────────────
    c.setFillColor(DARK)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(M, y, "¿Qué es?")
    y -= 5 * mm
    c.setFillColor(MID)
    c.setFont("Helvetica", 8.5)
    for line in [
        "Cada mañana lee automáticamente 14 diarios de Argentina,",
        "Uruguay y Paraguay, agrupa las noticias por evento y genera",
        "un resumen listo para leer o imprimir en menos de un minuto.",
    ]:
        c.drawString(M, y, line)
        y -= 4 * mm
    y -= 4 * mm

    # ── Hero screenshot (full content width) ─────────────────────────────────
    # remaining space: y  →  FTR_H + features_h + gap
    FEAT_H = 36 * mm
    GAP_AF = 6 * mm
    ss_max_h = y - FTR_H - FEAT_H - GAP_AF - 3 * mm
    dh = draw_img(c, crop1, M, y, INNER, ss_max_h)
    y -= dh + GAP_AF

    # ── Features 2×2 grid ─────────────────────────────────────────────────────
    feats = [
        ("01", "Resumen por país",
               "La noticia más importante del día en cada país"),
        ("02", "Historias del día",
               "Las noticias cubiertas por más de un diario, rankeadas"),
        ("03", "Distintas miradas",
               "Cómo cada medio tituló la misma historia"),
        ("04", "Listo para imprimir",
               "Abrís, imprimís (Ctrl+P), listo — A4 portrait"),
    ]
    col_w  = (INNER - 6 * mm) / 2
    row_h  = 15 * mm
    fy = y
    for i, (num, title, desc) in enumerate(feats):
        col  = i % 2
        row  = i // 2
        fx   = M + col * (col_w + 6 * mm)
        fy_i = fy - row * (row_h + 3 * mm)
        # light card background
        c.setFillColor(CREAM_DRK)
        c.roundRect(fx, fy_i - row_h, col_w, row_h, 1.5 * mm, fill=1, stroke=0)
        # number badge
        c.setFillColor(CRIMSON)
        c.roundRect(fx + 2 * mm, fy_i - row_h + (row_h - 5 * mm) / 2,
                    7 * mm, 5 * mm, 1 * mm, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(fx + 5.5 * mm, fy_i - row_h + (row_h - 5 * mm) / 2 + 1.5 * mm, num)
        # title
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(fx + 11 * mm, fy_i - 5 * mm, title)
        # desc
        c.setFillColor(MID)
        c.setFont("Helvetica", 7.5)
        c.drawString(fx + 11 * mm, fy_i - 9 * mm, desc)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — En Acción
# ══════════════════════════════════════════════════════════════════════════════

def make_qr(url):
    qr = qrcode.QRCode(version=2,
                       error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=10, border=3)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#003049", back_color="#FDF0D5")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


def page2(c, crop2, crop3, qr_url):
    DATA_H = 34 * mm

    cream_page(c)
    draw_header(c, "EN ACCIÓN", "Cono Sur Media Monitor")
    draw_footer(c,
        "Datos: Infobae · La Nación · Clarín · El Observador · ABC Color · y más",
        "Actualizado cada día")

    y = H - HDR_H - 5 * mm

    # ── Large screenshot (full width) ─────────────────────────────────────────
    dh1 = draw_img(c, crop2, M, y, INNER, 90 * mm)
    y -= dh1 + 5 * mm

    # ── Two screenshots side by side ──────────────────────────────────────────
    half_w = (INNER - 5 * mm) / 2
    dh2a = draw_img(c, crop3, M,             y, half_w, 68 * mm)
    dh2b = draw_img(c, crop2, M + half_w + 5 * mm, y, half_w, 68 * mm)
    y -= max(dh2a, dh2b) + 7 * mm

    # ── Features (left) + QR (right) ─────────────────────────────────────────
    QR_COL  = 42 * mm
    TXT_COL = INNER - QR_COL - 6 * mm

    # — Left: feature bullets —
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(M, y, "Lo que podés explorar")
    c.setStrokeColor(CRIMSON)
    c.setLineWidth(1.5)
    c.line(M, y - 1.5 * mm, M + 56 * mm, y - 1.5 * mm)
    by = y - 7 * mm
    bullets = [
        "Cliqueá cualquier noticia para leer la nota completa",
        "Compará cómo distintos diarios titularon el mismo evento",
        "Filtrá por país: Argentina, Uruguay o Paraguay",
        "Imprimí el briefing directo desde el navegador (Ctrl+P)",
        "Se actualiza solo, todos los días por la mañana",
    ]
    c.setFont("Helvetica", 8.5)
    c.setFillColor(MID)
    for b in bullets:
        c.drawString(M + 2 * mm, by, f"·  {b}")
        by -= 4.8 * mm

    # — Right: QR —
    qr_size = 34 * mm
    qr_x    = M + TXT_COL + 6 * mm
    qr_y    = y - qr_size - 5 * mm
    padding = 2.5 * mm
    c.setFillColor(CREAM_DRK)
    c.roundRect(qr_x - padding, qr_y - padding,
                qr_size + 2 * padding, qr_size + 2 * padding + 6 * mm,
                2 * mm, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 8)
    lbl = "Abrí el dashboard"
    lw  = c.stringWidth(lbl, "Helvetica-Bold", 8)
    c.drawString(qr_x + qr_size / 2 - lw / 2, qr_y + qr_size + 2.5 * mm, lbl)
    qr_ir = make_qr(qr_url)
    c.drawImage(qr_ir, qr_x, qr_y, qr_size, qr_size)
    c.setFillColor(MID)
    c.setFont("Helvetica", 6)
    uw = c.stringWidth(qr_url, "Helvetica", 6)
    c.drawString(qr_x + qr_size / 2 - uw / 2, qr_y - 3 * mm, qr_url)

    # ── DATA AT A GLANCE band ─────────────────────────────────────────────────
    band_y = FTR_H
    c.setFillColor(NAVY)
    c.rect(0, band_y, W, DATA_H, fill=1, stroke=0)

    c.setFillColor(BLUE_GREY)
    c.setFont("Helvetica-Bold", 7.5)
    label = "DATA AT A GLANCE"
    lw    = c.stringWidth(label, "Helvetica-Bold", 7.5)
    c.drawString((W - lw) / 2, band_y + DATA_H - 5.5 * mm, label)

    data_stats = [("14", "diarios"), ("3", "países"),
                  ("Diario", "actualización"), ("Gratis", "acceso libre")]
    col_w = W / len(data_stats)
    cy    = band_y + DATA_H / 2 - 3 * mm
    CARD_W, CARD_H = 34 * mm, 17 * mm
    for i, (num, lbl) in enumerate(data_stats):
        cx = col_w * i + col_w / 2
        c.setFillColor(HexColor("#00253A"))   # slightly darker navy for card
        c.roundRect(cx - CARD_W / 2, cy - CARD_H / 2, CARD_W, CARD_H,
                    2 * mm, fill=1, stroke=0)
        c.setFillColor(CREAM)
        c.setFont("Helvetica-Bold", 16)
        nw = c.stringWidth(num, "Helvetica-Bold", 16)
        c.drawString(cx - nw / 2, cy + 1.5 * mm, num)
        c.setFillColor(BLUE_GREY)
        c.setFont("Helvetica", 7)
        lw2 = c.stringWidth(lbl, "Helvetica", 7)
        c.drawString(cx - lw2 / 2, cy - 4.5 * mm, lbl)


# ══════════════════════════════════════════════════════════════════════════════
#  Build
# ══════════════════════════════════════════════════════════════════════════════

def build_ficha(site_url="auparrino.github.io/media-monitos"):
    crops = prepare_crops()
    crop1 = crops[0] if crops else None
    crop2 = crops[1] if crops else None
    crop3 = crops[2] if crops else None

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    c = canvas.Canvas(OUT, pagesize=A4)
    c.setTitle("Cono Sur Media Monitor — Ficha de proyecto")
    c.setAuthor("Augusto")

    page1(c, crop1)
    c.showPage()
    page2(c, crop2, crop3, site_url)
    c.showPage()
    c.save()
    print(f"Ficha guardada: {OUT}")


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "auparrino.github.io/media-monitos"
    build_ficha(url)
