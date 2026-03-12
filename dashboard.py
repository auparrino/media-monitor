"""Generate policy briefing document — story-based, print-ready."""

import os
import re
import json
import sqlite3
from datetime import datetime, timedelta
from collections import Counter

import requests as http_requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "news.db")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "dashboard.html")

CATEGORY_COLORS = {
    "politica": "#780000",
    "economia": "#C1121F",
    "internacional": "#669BBC",
}
CATEGORY_LABELS = {
    "politica": "POL",
    "economia": "ECON",
    "internacional": "INTL",
}

_TWEMOJI = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72"
COUNTRY_FLAGS = {
    "argentina": f'<img class="flag" src="{_TWEMOJI}/1f1e6-1f1f7.png" alt="AR">',
    "uruguay": f'<img class="flag" src="{_TWEMOJI}/1f1fa-1f1fe.png" alt="UY">',
    "paraguay": f'<img class="flag" src="{_TWEMOJI}/1f1f5-1f1fe.png" alt="PY">',
}
COUNTRY_NAMES = {"argentina": "Argentina", "uruguay": "Uruguay", "paraguay": "Paraguay"}
COUNTRIES = ["argentina", "uruguay", "paraguay"]

# Daily noise patterns — routine articles that always cluster but add no insight
NOISE_PATTERNS = [
    r"d[oó]lar hoy",
    r"d[oó]lar blue hoy",
    r"d[oó]lar ccl hoy",
    r"cotizaci[oó]n del? .*d[oó]lar",
    r"cotizaci[oó]n del? .*miércoles|lunes|martes|jueves|viernes|sábado|domingo",
    r"d[oó]lar.*a cu[aá]nto cotiza",
    r"esta es la cotizaci[oó]n",
    r"tipo de cambio hoy",
    r"precio del d[oó]lar hoy",
    r"clima hoy",
    r"hor[oó]scopo",
    r"efem[eé]rides",
    # Ads, promos, lifestyle junk from RSS feeds
    r"suscri[bp]",
    r"comunidad de suscriptores",
    r"vodka|gin tonic|cocktail",
    r"receta de ",
    r"desaf[ií]o moos",
    r"colegio biling[uü]e",
    r"en vivo:?\s*(pe[nñ]arol|nacional|racing|boca|river|independiente)",
    r"vs\.\s.*(en vivo|en directo)",
    r"(apertura|clausura).*en vivo",
]

STOPWORDS = {
    "de", "la", "el", "en", "que", "y", "a", "los", "las", "del", "un", "una",
    "por", "con", "para", "se", "su", "al", "es", "lo", "no", "más", "mas",
    "como", "pero", "sus", "le", "ya", "entre", "sin", "sobre", "este", "ser",
    "son", "también", "tambien", "fue", "ha", "han", "hay", "muy", "tras",
    "desde", "hasta", "ante", "según", "segun", "puede", "nuevo", "nueva",
    "nuevos", "nuevas", "dos", "tres", "primer", "primera", "cada", "otro",
    "otra", "otros", "todas", "todos", "todo", "toda", "estos", "estas",
    "ese", "esa", "esos", "esas", "será", "seria", "donde", "parte", "solo",
    "así", "asi", "mismo", "misma", "día", "dia", "hoy", "año", "anos",
    "años", "mes", "semana", "contra", "luego", "después", "despues",
    "mientras", "dijo", "afirmó", "señaló", "explicó", "aseguró", "indicó",
    "agregó",
}


def load_articles():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM articles ORDER BY published_at DESC", conn)
    conn.close()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["published_at"] = df["published_at"].dt.tz_localize(None)
    df["date"] = df["published_at"].dt.date
    return df


# ── Story Clustering ────────────────────────────────────────────────────

def _normalize(title):
    t = title.lower()
    t = re.sub(r'[^a-záéíóúñü\s]', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def _keywords(title):
    words = re.findall(r'[a-záéíóúñü]+', title.lower())
    return set(w for w in words if len(w) >= 4 and w not in STOPWORDS)


def _is_noise(title):
    """Check if a title matches daily noise patterns, ads, or junk content."""
    t = title.lower()
    # Too short to be a real headline
    if len(t) < 25:
        return True
    return any(re.search(p, t) for p in NOISE_PATTERNS)


_TITLE_PROMPT = (
    "Sos un editor de un briefing diplomático. "
    "A partir de los siguientes titulares que cubren el mismo evento, "
    "generá UN SOLO título síntesis, factual y conciso (máximo 100 caracteres). "
    "No uses comillas, no uses 'Quién es', no uses 'EN VIVO'. "
    "Solo respondé con el título, nada más."
)


def _llm_synthesize_title(headlines):
    """Call LLM API to synthesize a title. Tries Groq first, then Gemini."""
    bullet_list = "\n".join(f"- {h}" for h in headlines)
    user_msg = f"Titulares:\n{bullet_list}"

    # Try Groq (Llama 3)
    if GROQ_API_KEY:
        try:
            resp = http_requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": _TITLE_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 80,
                },
                timeout=10,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            text = text.strip('"\'""«»\n .')
            if 5 < len(text) < 130:
                return text
        except Exception:
            pass

    # Try Gemini
    if GEMINI_API_KEY:
        try:
            resp = http_requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": f"{_TITLE_PROMPT}\n\n{user_msg}"}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 80},
                },
                timeout=10,
            )
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            text = text.strip('"\'""«»\n .')
            if 5 < len(text) < 130:
                return text
        except Exception:
            pass

    return None


def _generate_story_title(articles):
    """Generate a synthesized story title.

    Tries Gemini Flash first for true LLM synthesis.
    Falls back to TF-IDF centroid selection if API unavailable.
    """
    if len(articles) == 1:
        return articles[0].title

    titles = list(dict.fromkeys(art.title for art in articles))  # dedupe preserving order

    # Try LLM synthesis
    llm_title = _llm_synthesize_title(titles[:6])  # max 6 headlines to keep prompt small
    if llm_title:
        return llm_title

    # Fallback: TF-IDF centroid-based selection
    clean = []
    for t in titles:
        t = re.sub(r',?\s*EN VIVO:?\s*', ': ', t, flags=re.IGNORECASE)
        t = re.sub(r'^:\s*', '', t)
        t = re.sub(r'\s*\|.*$', '', t)
        t = re.sub(r'\s+', ' ', t).strip()
        clean.append(t)

    norms = [_normalize(t) for t in clean]
    vectorizer = TfidfVectorizer(
        token_pattern=r"[a-záéíóúñü]{3,}",
        stop_words=list(STOPWORDS),
    )
    try:
        tfidf = vectorizer.fit_transform(norms)
    except ValueError:
        return clean[0]

    centroid = np.asarray(tfidf.mean(axis=0))
    sims = cosine_similarity(tfidf, centroid).flatten()
    best_idx = int(sims.argmax())
    base = clean[best_idx]

    if len(base) > 95:
        cut = base[:95].rfind(' ')
        if cut > 40:
            base = base[:cut] + "..."

    return base


def _avg_linkage_cluster(titles_norm, kw_sets, threshold, min_kw_overlap):
    """Run average-linkage clustering on a set of normalised titles.

    Returns list of index-groups (each group is a list of ints indexing
    into the input lists).
    """
    from collections import defaultdict
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n = len(titles_norm)
    if n < 2:
        return [list(range(n))]

    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[a-záéíóúñü]{3,}",
        stop_words=list(STOPWORDS),
        sublinear_tf=True,
    )
    try:
        tfidf = vec.fit_transform(titles_norm)
    except ValueError:
        return [list(range(n))]

    sim = cosine_similarity(tfidf)
    dist = np.clip(1.0 - sim, 0, 2.0)
    np.fill_diagonal(dist, 0)

    for i in range(n):
        for j in range(i + 1, n):
            if len(kw_sets[i] & kw_sets[j]) < min_kw_overlap:
                dist[i, j] = 1.0
                dist[j, i] = 1.0

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, t=1.0 - threshold, criterion='distance')

    groups = defaultdict(list)
    for i, lbl in enumerate(labels):
        groups[lbl].append(i)
    return list(groups.values())


def cluster_stories(df):
    """Two-pass clustering to group articles by event.

    Pass 1 — Liberal grouping (threshold 0.25): merges articles that
    share the same broad topic so related framings don't get split.

    Pass 2 — Strict re-clustering (threshold 0.35): re-vectorises each
    large cluster in isolation.  Because TF-IDF is re-fit on just those
    articles, generic topic words (guerra, irán) lose weight and sub-event
    keywords (OTAN, misil, petróleo) gain weight, naturally splitting
    mega-clusters into precise sub-events.
    """
    PASS1_THRESHOLD = 0.20
    PASS2_THRESHOLD = 0.40
    RECLUSTER_MIN = 10   # only split really large clusters
    MIN_KW = 2

    rows = list(df.itertuples())
    if not rows:
        return []

    titles_norm = [_normalize(r.title) for r in rows]
    kw_sets = [_keywords(r.title) for r in rows]

    # ── Pass 1: broad grouping ──
    pass1_groups = _avg_linkage_cluster(titles_norm, kw_sets, PASS1_THRESHOLD, MIN_KW)

    # ── Pass 2: refine large clusters ──
    final_groups = []
    for group in pass1_groups:
        if len(group) < RECLUSTER_MIN:
            final_groups.append(group)
            continue

        # Re-vectorise only this cluster's articles for sharper IDF
        sub_titles = [titles_norm[i] for i in group]
        sub_kw = [kw_sets[i] for i in group]
        sub_groups = _avg_linkage_cluster(sub_titles, sub_kw, PASS2_THRESHOLD, MIN_KW)

        # Map local indices back to global
        for sg in sub_groups:
            final_groups.append([group[i] for i in sg])

    # ── Build cluster objects ──
    clusters = []
    for indices in final_groups:
        cluster_rows = [rows[k] for k in indices]
        sources = sorted(set(r.source for r in cluster_rows))
        countries = sorted(set(r.country for r in cluster_rows))
        categories = sorted(set(r.category for r in cluster_rows))

        noise_count = sum(1 for r in cluster_rows if _is_noise(r.title))
        is_noise = noise_count > len(cluster_rows) * 0.5

        clusters.append({
            "articles": cluster_rows,
            "sources": sources,
            "countries": countries,
            "categories": categories,
            "lead": cluster_rows[0].title,
            "title": _generate_story_title(cluster_rows),
            "size": len(cluster_rows),
            "multi": len(sources) >= 2,
            "noise": is_noise,
            "indices": set(indices),
        })

    clusters.sort(key=lambda c: (c["multi"], len(c["sources"]), c["size"]), reverse=True)
    return clusters


# ── HTML Builders ───────────────────────────────────────────────────────

def build_overview(clusters, df):
    """One-line overview per country from top stories.

    Avoids repeating the same international story for multiple countries.
    Prefers stories with domestic sources (not just wire/international).
    """
    lines = {}
    used_titles = set()

    for country in COUNTRIES:
        for c in clusters:
            if country not in c["countries"] or not c["multi"] or c["noise"]:
                continue
            # Skip if we already used this exact story for another country
            if c["title"] in used_titles:
                continue
            # Prefer stories that have a domestic source for this country
            has_domestic = any(
                r.country == country for r in c["articles"]
            )
            if has_domestic:
                lines[country] = c["title"]
                used_titles.add(c["title"])
                break

        if country not in lines:
            # Fallback: first non-noise article from this country
            cdf = df[df["country"] == country]
            for _, row in cdf.iterrows():
                if not _is_noise(row["title"]) and row["title"] not in used_titles:
                    lines[country] = row["title"]
                    used_titles.add(row["title"])
                    break
            if country not in lines:
                lines[country] = "Sin cobertura reciente."

    items = ""
    for country in COUNTRIES:
        flag = COUNTRY_FLAGS[country]
        name = COUNTRY_NAMES[country]
        items += f'<li><strong>{flag} {name}:</strong> {lines[country]}</li>\n'
    return f'<ul class="overview-list">{items}</ul>'


def build_stories(clusters):
    """Key stories section — clustered headlines showing editorial framing."""
    multi = [c for c in clusters if c["multi"] and not c["noise"]]
    if not multi:
        return '<p class="muted">No multi-source stories identified in this cycle.</p>'

    html = ""
    for i, c in enumerate(multi[:12], 1):
        # Category badges
        badges = ""
        for cat in c["categories"]:
            color = CATEGORY_COLORS.get(cat, "#999")
            label = CATEGORY_LABELS.get(cat, cat)
            badges += f'<span class="badge" style="background:{color}">{label}</span> '

        # Country flags
        flags = " ".join(COUNTRY_FLAGS.get(co, "") for co in c["countries"])

        # Source-by-source headlines with links
        source_lines = ""
        seen = set()
        for art in c["articles"]:
            if art.source in seen:
                continue
            seen.add(art.source)
            source_lines += f"""
            <div class="story-variant">
                <span class="sv-source">{art.source}</span>
                <a class="sv-title" href="{art.url}" target="_blank">{art.title}</a>
            </div>"""

        html += f"""
        <div class="story">
            <div class="story-header">
                <span class="story-num">{i}</span>
                {badges} {flags}
                <span class="story-sources">{len(c['sources'])} fuentes</span>
            </div>
            <div class="story-title">{c['title']}</div>
            <div class="story-body">{source_lines}</div>
        </div>"""

    return html


def build_also_reported(df, clusters, country, limit=10):
    """Headlines NOT part of any multi-source cluster — 'also reported' items.

    Round-robins across sources so no single outlet dominates the list.
    """
    # Collect indices used in multi-source clusters
    used = set()
    for c in clusters:
        if c["multi"]:
            used |= c["indices"]

    cdf = df[df["country"] == country]

    # Group remaining articles by source
    by_source = {}
    for idx, row in cdf.iterrows():
        if idx not in used and not _is_noise(row["title"]):
            src = row["source"]
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(row)

    # Round-robin pick across sources
    remaining = []
    source_iters = {s: iter(rows) for s, rows in by_source.items()}
    while len(remaining) < limit and source_iters:
        exhausted = []
        for src in list(source_iters.keys()):
            if len(remaining) >= limit:
                break
            try:
                remaining.append(next(source_iters[src]))
            except StopIteration:
                exhausted.append(src)
        for src in exhausted:
            del source_iters[src]

    if not remaining:
        return ""

    rows_html = ""
    for row in remaining:
        cat = row["category"]
        color = CATEGORY_COLORS.get(cat, "#999")
        label = CATEGORY_LABELS.get(cat, cat)
        date_str = row["published_at"].strftime("%d/%m") if pd.notna(row["published_at"]) else ""
        rows_html += f"""
        <div class="also-item">
            <span class="also-date">{date_str}</span>
            <span class="badge" style="background:{color}">{label}</span>
            <span class="also-source">{row['source']}</span>
            <a class="also-title" href="{row['url']}" target="_blank">{row['title']}</a>
        </div>"""

    flag = COUNTRY_FLAGS[country]
    name = COUNTRY_NAMES[country]
    return f"""
    <div class="also-section">
        <div class="also-header">{flag} {name} &mdash; Also Reported</div>
        {rows_html}
    </div>"""



# ── Main ────────────────────────────────────────────────────────────────

def generate_dashboard():
    df = load_articles()
    now_str = datetime.now().strftime("%d %B %Y, %H:%M")
    now_short = datetime.now().strftime("%Y-%m-%d %H:%M")

    clusters = cluster_stories(df)
    n_multi = len([c for c in clusters if c["multi"] and not c["noise"]])

    overview = build_overview(clusters, df)
    stories = build_stories(clusters)
    also_reported = "".join(build_also_reported(df, clusters, c) for c in COUNTRIES)

    total = len(df)
    sources = df["source"].nunique()

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cono Sur Policy Monitor &mdash; {now_short}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    font-family: 'Segoe UI', Arial, Helvetica, sans-serif;
    background: #FDF0D5;
    color: #222;
    line-height: 1.55;
    font-size: 11.5px;
    max-width: 800px;
    margin: 0 auto;
    padding: 0;
}}

/* ── Letterhead ── */
.letterhead {{
    background: #003049;
    padding: 22px 32px 16px;
    color: white;
}}
.lh-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.lh-title {{
    font-size: 20px;
    font-weight: 800;
    color: #FDF0D5;
    letter-spacing: 2.5px;
}}
.lh-sub {{
    font-size: 11px;
    color: #669BBC;
    font-weight: 600;
    letter-spacing: 1px;
    margin-top: 2px;
}}
.lh-date {{
    font-size: 11px;
    color: #669BBC;
    text-align: right;
}}

/* ── Content ── */
.content {{
    padding: 14px 32px 16px;
    background: white;
}}

/* ── Section headers ── */
.sec-head {{
    font-size: 11px;
    font-weight: 700;
    color: #003049;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 18px 0 8px;
    padding-bottom: 4px;
    border-bottom: 2px solid #C1121F;
}}
.sec-head:first-child {{ margin-top: 8px; }}
.sec-desc {{
    font-size: 11px;
    color: #555;
    margin: 2px 0 10px;
    padding: 6px 10px;
    background: #f8f4ec;
    border-radius: 3px;
    line-height: 1.4;
}}

/* ── Overview ── */
.overview-list {{
    list-style: none;
    margin: 0;
    padding: 0;
}}
.overview-list li {{
    padding: 5px 8px;
    font-size: 12px;
    line-height: 1.4;
    border-bottom: 1px solid #f0e8d8;
}}
.overview-list li:last-child {{ border-bottom: none; }}

/* ── Stories ── */
.story {{
    margin-bottom: 14px;
    border-left: 3px solid #C1121F;
    padding-left: 12px;
}}
.story-header {{
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
    flex-wrap: wrap;
}}
.story-num {{
    background: #003049;
    color: #FDF0D5;
    width: 18px; height: 18px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 700;
    flex-shrink: 0;
}}
.story-sources {{
    font-size: 9.5px;
    color: #999;
    margin-left: auto;
}}
.story-title {{
    font-size: 13px;
    font-weight: 700;
    color: #003049;
    line-height: 1.35;
    margin-bottom: 4px;
    padding-bottom: 4px;
    border-bottom: 1px dotted #ddd;
}}
.story-body {{
    padding: 2px 0;
}}
.story-variant {{
    display: flex;
    gap: 8px;
    padding: 2px 0;
    font-size: 11.5px;
    line-height: 1.35;
    align-items: baseline;
}}
.sv-source {{
    font-weight: 700;
    color: #780000;
    font-size: 10px;
    min-width: 90px;
    flex-shrink: 0;
}}
.sv-title {{
    flex: 1;
    color: #222;
    text-decoration: none;
}}
.sv-title:hover {{
    text-decoration: underline;
}}

/* ── Badge ── */
.badge {{
    display: inline-block;
    padding: 1px 5px;
    border-radius: 2px;
    color: white;
    font-size: 8px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    white-space: nowrap;
    vertical-align: middle;
}}

/* ── Also Reported ── */
.also-section {{
    margin-bottom: 12px;
}}
.also-header {{
    font-size: 11px;
    font-weight: 700;
    color: #003049;
    padding: 4px 0;
    border-bottom: 1px solid #e0d8c8;
    margin-bottom: 3px;
}}
.also-item {{
    display: flex;
    gap: 6px;
    padding: 2px 0;
    font-size: 11px;
    line-height: 1.35;
    align-items: baseline;
    border-bottom: 1px dotted #ece4d4;
}}
.also-item:last-child {{ border-bottom: none; }}
.also-date {{ color: #999; font-size: 9.5px; min-width: 32px; }}
.also-source {{ font-weight: 600; color: #780000; font-size: 9.5px; min-width: 85px; flex-shrink: 0; }}
.also-title {{ flex: 1; color: #222; text-decoration: none; }}
.also-title:hover {{ text-decoration: underline; }}

/* ── Footer ── */
.doc-footer {{
    background: #003049;
    padding: 10px 32px;
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    color: #669BBC;
}}

.muted {{ color: #999; font-style: italic; }}
a {{ color: inherit; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.flag {{ width: 16px; height: 16px; vertical-align: middle; display: inline-block; }}

/* ── Print ── */
@media print {{
    body {{ font-size: 10.5px; max-width: none; background: white; }}
    .letterhead {{ padding: 14px 20px 10px; }}
    .content {{ padding: 8px 20px 10px; }}
    .story {{ break-inside: avoid; }}
    .also-section {{ break-inside: avoid; }}
    .letterhead, .badge, .story-num, .doc-footer {{
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }}
    @page {{ margin: 1cm; size: A4 portrait; }}
}}
</style>
</head>
<body>

<!-- Letterhead -->
<div class="letterhead">
    <div class="lh-top">
        <div>
            <div class="lh-title">CONO SUR POLICY MONITOR</div>
            <div class="lh-sub">{COUNTRY_FLAGS['argentina']} Argentina &middot; {COUNTRY_FLAGS['uruguay']} Uruguay &middot; {COUNTRY_FLAGS['paraguay']} Paraguay</div>
        </div>
        <div class="lh-date">{now_str}</div>
    </div>
</div>

<div class="content">

    <div class="sec-head">1. Overview</div>
    <p class="sec-desc">The most important story of the day for each country at a glance.</p>
    {overview}

    <div class="sec-head">2. Key Stories ({n_multi} multi-source stories)</div>
    <p class="sec-desc">News events picked up by multiple outlets. The more sources covering a story, the higher it ranks. Each entry shows how different newsrooms framed the same event.</p>
    {stories}

    <div class="sec-head">3. Also Reported</div>
    <p class="sec-desc">Other noteworthy headlines from individual outlets that didn't appear in multiple sources.</p>
    {also_reported}

</div>

<div class="doc-footer">
    <span>CONO SUR POLICY MONITOR</span>
    <span>{now_short} | {total} articles, {sources} sources</span>
</div>

</body>
</html>"""

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"   Dashboard generado: {OUTPUT_PATH}")
    print(f"      {total} articulos, {sources} fuentes, {n_multi} multi-source stories")


if __name__ == "__main__":
    generate_dashboard()
