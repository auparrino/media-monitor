"""
Cono Sur Media Monitor — 2-page project ficha.
Style mirrors politicdash_ficha_v3.pdf:
  Page 1 — overview, description, features, screenshot
  Page 2 — IN ACTION (screenshots grid), feature list, DATA AT A GLANCE band
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT = os.path.join(BASE, "output", "_dashboard_screenshot.png")
OUTPUT    = os.path.join(BASE, "output", "ficha_conosur_monitor.pdf")

# ── Palette (same as dashboard) ───────────────────────────────────────────────
NAVY      = HexColor("#003049")
CRIMSON   = HexColor("#C1121F")
CREAM     = HexColor("#FDF0D5")
BLUE_GREY = HexColor("#669BBC")
WHITE     = HexColor("#FFFFFF")
MID       = HexColor("#555555")
DARK      = HexColor("#222222")
LIGHT_BG  = HexColor("#F5F0E8")
SHADOW    = HexColor("#E0D8C8")

W, H = A4   # 595.28 × 841.89 pt
MARGIN = 17 * mm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _header_bar(c, title_left, subtitle_left, label_right, y_top, height=14*mm):
    """Navy header bar with left title + right label."""
    c.setFillColor(NAVY)
    c.rect(0, y_top - height, W, height, fill=1, stroke=0)
    # left title
    c.setFillColor(CREAM)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(MARGIN, y_top - height + 4.5*mm, title_left)
    # left subtitle
    if subtitle_left:
        c.setFillColor(BLUE_GREY)
        c.setFont("Helvetica", 8)
        tw = c.stringWidth(title_left, "Helvetica-Bold", 15)
        c.drawString(MARGIN + tw + 5*mm, y_top - height + 5*mm, subtitle_left)
    # right label
    if label_right:
        c.setFillColor(BLUE_GREY)
        c.setFont("Helvetica", 8)
        c.drawRightString(W - MARGIN, y_top - height + 5*mm, label_right)
    # crimson accent line below header
    c.setStrokeColor(CRIMSON)
    c.setLineWidth(2)
    c.line(0, y_top - height - 1*mm, W, y_top - height - 1*mm)


def _footer_bar(c, left_text, right_text, height=8*mm):
    c.setFillColor(NAVY)
    c.rect(0, 0, W, height, fill=1, stroke=0)
    c.setFillColor(BLUE_GREY)
    c.setFont("Helvetica", 7)
    c.drawString(MARGIN, 2.6*mm, left_text)
    c.drawRightString(W - MARGIN, 2.6*mm, right_text)


def _place_image(c, path, x, y, max_w, max_h, shadow=True):
    """Draw image scaled to fit max_w × max_h, return actual height drawn."""
    if not os.path.exists(path):
        return 0
    img = ImageReader(path)
    iw, ih = img.getSize()
    scale = min(max_w / iw, max_h / ih)
    dw, dh = iw * scale, ih * scale
    if shadow:
        c.setFillColor(SHADOW)
        c.rect(x + 1.5, y - dh - 1.5, dw, dh, fill=1, stroke=0)
    c.drawImage(path, x, y - dh, dw, dh)
    c.setStrokeColor(HexColor("#CCCCCC"))
    c.setLineWidth(0.4)
    c.rect(x, y - dh, dw, dh, fill=0, stroke=1)
    return dh


def _stat_block(c, x, y, number, label, num_size=28, lbl_size=8):
    """Big number + small label centered at (x, y)."""
    c.setFont("Helvetica-Bold", num_size)
    c.setFillColor(CREAM)
    nw = c.stringWidth(number, "Helvetica-Bold", num_size)
    c.drawString(x - nw/2, y, number)
    c.setFont("Helvetica", lbl_size)
    c.setFillColor(BLUE_GREY)
    lw = c.stringWidth(label, "Helvetica", lbl_size)
    c.drawString(x - lw/2, y - lbl_size*0.9, label)


def _bullet(c, x, y, text, font_size=9, color=None):
    c.setFont("Helvetica", font_size)
    c.setFillColor(color or MID)
    c.drawString(x, y, f"\u2022  {text}")
    return y - font_size * 1.55


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

def page1(c, site_url):
    HEADER_H = 14 * mm
    FOOTER_H = 8 * mm

    # ── Header ──
    _header_bar(c,
                "CONO SUR MEDIA MONITOR",
                "Daily Intelligence Briefing  |  Argentina \u00b7 Uruguay \u00b7 Paraguay",
                "Personal Project | 2025\u20132026",
                H, HEADER_H)

    # ── Tag row ──
    y = H - HEADER_H - 3*mm
    c.setFillColor(CRIMSON)
    c.rect(MARGIN, y - 5*mm, 28*mm, 5*mm, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(MARGIN + 2*mm, y - 3.8*mm, "CONO SUR")

    c.setFillColor(DARK)
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN + 31*mm, y - 3.8*mm, "Personal Project  |  2025\u20132026")

    # ── Main title ──
    y -= 9*mm
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(MARGIN, y, "MEDIA MONITOR")

    # ── URL ──
    y -= 6*mm
    c.setFillColor(BLUE_GREY)
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN, y, site_url)

    # ── Stats row ──
    y -= 8*mm
    stats = [
        ("14",   "news sources"),
        ("3",    "countries"),
        ("500+", "articles/day"),
    ]
    c.setFillColor(NAVY)
    col_w = (W - 2*MARGIN) / len(stats)
    for i, (num, lbl) in enumerate(stats):
        cx = MARGIN + col_w * i + col_w / 2
        c.setFont("Helvetica-Bold", 22)
        c.setFillColor(NAVY)
        nw = c.stringWidth(num, "Helvetica-Bold", 22)
        c.drawString(cx - nw/2, y, num)
        c.setFont("Helvetica", 8)
        c.setFillColor(MID)
        lw = c.stringWidth(lbl, "Helvetica", 8)
        c.drawString(cx - lw/2, y - 4*mm, lbl)

    # thin separator
    y -= 9*mm
    c.setStrokeColor(HexColor("#DDDDDD"))
    c.setLineWidth(0.5)
    c.line(MARGIN, y, W - MARGIN, y)
    y -= 4*mm

    # ── Description ──
    usable_w = W - 2*MARGIN
    LEFT_W  = usable_w * 0.55   # description + features
    RIGHT_X = MARGIN + LEFT_W + 5*mm
    RIGHT_W = usable_w - LEFT_W - 5*mm
    desc_y  = y

    c.setFillColor(DARK)
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(MARGIN, y,
                 "Automated daily news briefing for the Southern Cone")
    y -= 5*mm

    c.setFont("Helvetica", 8.5)
    c.setFillColor(MID)
    lines = [
        "Scrapes 14 newspapers across Argentina, Uruguay and Paraguay",
        "every day — then groups articles by event, synthesises headlines",
        "with AI, and produces a print-ready briefing document.",
    ]
    for line in lines:
        c.drawString(MARGIN, y, line)
        y -= 3.8*mm

    # ── Feature sections ──
    y -= 5*mm

    features = [
        ("Daily scraping",
         "14 RSS + HTML sources across 3 countries. Automatic fallback\n"
         "to HTML parsing when RSS feeds are unavailable."),
        ("2-pass ML clustering",
         "Average-linkage TF-IDF clustering groups articles by event.\n"
         "A strict second pass splits large clusters into sub-events."),
        ("AI title synthesis",
         "Groq (Llama 3) and Gemini Flash synthesise a single headline\n"
         "from multiple source framings of the same story."),
        ("Print-ready briefing",
         "Overview, ranked key stories, and 'also reported' sections\n"
         "in one self-contained HTML file — print directly from browser."),
    ]

    for title, desc in features:
        # section label
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(MARGIN, y, title)
        y -= 3.5*mm
        # crimson underline
        tw = c.stringWidth(title, "Helvetica-Bold", 9)
        c.setStrokeColor(CRIMSON)
        c.setLineWidth(1)
        c.line(MARGIN, y + 1*mm, MARGIN + tw, y + 1*mm)
        # description (two lines)
        c.setFillColor(MID)
        c.setFont("Helvetica", 8)
        for dline in desc.split("\n"):
            c.drawString(MARGIN + 2*mm, y, dline)
            y -= 3.5*mm
        y -= 2*mm

    # ── Screenshot (right column) ──
    ss_top  = desc_y
    ss_h    = ss_top - FOOTER_H - 6*mm
    _place_image(c, SCREENSHOT, RIGHT_X, ss_top, RIGHT_W, ss_h)

    # ── CTA ──
    cta_y = FOOTER_H + 14*mm
    c.setFillColor(NAVY)
    c.roundRect(MARGIN, cta_y, 55*mm, 8*mm, 2*mm, fill=1, stroke=0)
    c.setFillColor(CREAM)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawCentredString(MARGIN + 27.5*mm, cta_y + 2.8*mm, "View live dashboard")

    c.setFillColor(MID)
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN + 58*mm, cta_y + 2.8*mm, site_url)

    c.setFillColor(HexColor("#AAAAAA"))
    c.setFont("Helvetica", 7.5)
    c.drawString(MARGIN, FOOTER_H + 5*mm, "Works on desktop and tablet \u00b7 updated daily by GitHub Actions")

    # ── Footer ──
    _footer_bar(c,
                "Cono Sur Media Monitor",
                "Python \u00b7 scikit-learn \u00b7 Groq API \u00b7 Gemini API \u00b7 pandas \u00b7 SQLite")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — IN ACTION
# ══════════════════════════════════════════════════════════════════════════════

def page2(c, site_url):
    FOOTER_H = 8 * mm
    DATA_BAND_H = 32 * mm

    # ── Header ──
    _header_bar(c,
                "IN ACTION",
                "",
                "Cono Sur Media Monitor",
                H, 14*mm)

    y = H - 14*mm - 5*mm

    # ── Screenshot (large, centre) ──
    ss_top = y
    ss_h   = 80 * mm
    ss_w   = W - 2*MARGIN
    dh = _place_image(c, SCREENSHOT, MARGIN, ss_top, ss_w, ss_h)
    y = ss_top - dh - 8*mm

    # ── Two-column layout: features left, URL/note right ──
    COL_W = (W - 2*MARGIN - 6*mm) / 2
    LEFT_X  = MARGIN
    RIGHT_X = MARGIN + COL_W + 6*mm

    # Left — "What you can explore"
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(LEFT_X, y, "What you can explore")
    c.setStrokeColor(CRIMSON)
    c.setLineWidth(1)
    c.line(LEFT_X, y - 1.5*mm, LEFT_X + 50*mm, y - 1.5*mm)
    y -= 5*mm

    bullets = [
        "Browse today's top stories ranked by source count",
        "See how different outlets frame the same event",
        "Filter by country: Argentina, Uruguay or Paraguay",
        "Read the full 'Also Reported' section for each country",
        "Print the briefing directly from your browser (Ctrl+P)",
    ]
    feat_y = y
    for b in bullets:
        feat_y = _bullet(c, LEFT_X + 1*mm, feat_y, b, font_size=8.5)

    # Right — URL + stack
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(RIGHT_X, y, "Live at")
    c.setFillColor(CRIMSON)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(RIGHT_X + 13*mm, y, site_url)

    c.setFillColor(MID)
    c.setFont("Helvetica", 8)
    stack_y = y - 6*mm
    stack = [
        "Python 3 \u00b7 requests \u00b7 feedparser",
        "scikit-learn \u00b7 scipy \u00b7 TF-IDF",
        "Groq Llama-3 \u00b7 Gemini Flash",
        "SQLite \u00b7 pandas \u00b7 ReportLab",
        "GitHub Actions (daily update)",
    ]
    for s in stack:
        c.drawString(RIGHT_X, stack_y, s)
        stack_y -= 3.8*mm

    # ── DATA AT A GLANCE band ──
    band_y = FOOTER_H + DATA_BAND_H
    c.setFillColor(NAVY)
    c.rect(0, FOOTER_H, W, DATA_BAND_H, fill=1, stroke=0)

    # Label
    c.setFillColor(BLUE_GREY)
    c.setFont("Helvetica-Bold", 9)
    lw = c.stringWidth("DATA AT A GLANCE", "Helvetica-Bold", 9)
    c.drawString((W - lw)/2, FOOTER_H + DATA_BAND_H - 6*mm, "DATA AT A GLANCE")

    # 4 stat blocks
    data_stats = [
        ("14",   "news sources"),
        ("3",    "countries\ncovered"),
        ("500+", "articles\nper day"),
        ("AI",   "headline\nsynthesis"),
    ]
    col_w = W / len(data_stats)
    num_y = FOOTER_H + DATA_BAND_H - 16*mm
    for i, (num, lbl) in enumerate(data_stats):
        cx = col_w * i + col_w / 2
        _stat_block(c, cx, num_y, num, lbl.replace("\n", " "), num_size=26, lbl_size=8)

    # Tagline
    c.setFillColor(BLUE_GREY)
    c.setFont("Helvetica", 7.5)
    tagline = f"Fully automated \u00b7 open source \u00b7 {site_url}"
    tw = c.stringWidth(tagline, "Helvetica", 7.5)
    c.drawString((W - tw)/2, FOOTER_H + 2.5*mm, tagline)

    # ── Footer ──
    _footer_bar(c,
                "Cono Sur Media Monitor",
                "Data: Infobae \u00b7 La Nacion \u00b7 Clarin \u00b7 El Observador \u00b7 ABC Color + 9 more")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def build_ficha(site_url="github.com/YOUR-USER/media-monitos"):
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    c = canvas.Canvas(OUTPUT, pagesize=A4)
    c.setTitle("Cono Sur Media Monitor — Project Ficha")
    c.setAuthor("Augusto")

    page1(c, site_url)
    c.showPage()
    page2(c, site_url)
    c.save()
    print(f"Ficha saved: {OUTPUT}")


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "YOUR-USER.github.io/media-monitos"
    build_ficha(url)
