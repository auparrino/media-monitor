"""Generate policy briefing document — story-based, print-ready."""

import html as html_mod
import os
import re
import json
import sqlite3
import time
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import requests as http_requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

from categorizer import analyze_article
from name_utils import (
    looks_like_person_name as _looks_like_person_name,
    normalize_person_name as _normalize_person_name,
    NON_PERSON_TOKENS as _NON_PERSON_NAME_TOKENS,
    NAME_CONNECTORS as _PERSON_CONNECTORS,
)


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "news.db")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "dashboard.html")

# ── Rate-limit and delay constants ────────────────────────────────────
LLM_CALL_DELAY = 2          # seconds between LLM calls
LLM_RETRY_COOLDOWN = 10     # seconds before retrying failed briefs
LLM_RETRY_DELAY = 3         # seconds between retry LLM calls


def _esc(text):
    """HTML-escape user-supplied text to prevent XSS."""
    return html_mod.escape(str(text)) if text else ""

CATEGORY_COLORS = {
    "politica": "#7d3c98",
    "economia": "#d4a800",
    "internacional": "#1a6fa3",
}
CATEGORY_LABELS = {
    "politica": "POL",
    "economia": "ECON",
    "internacional": "INTL",
}
SUBCATEGORY_LABELS = {
    "macroeconomia": "Macro",
    "energia": "Energia",
    "agro": "Agro",
    "comercio": "Comercio",
    "obra_publica": "Obra Publica",
    "empleo": "Empleo",
    "ejecutivo": "Ejecutivo",
    "legislativo": "Legislativo",
    "electoral": "Electoral",
    "justicia": "Justicia",
    "seguridad": "Seguridad",
    "social": "Social",
    "diplomacia": "Diplomacia",
    "comercio_exterior": "Comercio Ext.",
    "conflictos": "Conflictos",
    "multilateral": "Multilateral",
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
    r"cu[aá]nto cuesta (?:disney|netflix|spotify|hbo|amazon)",
    r"colegio biling[uü]e",
    r"en vivo:?\s*(pe[nñ]arol|nacional|racing|boca|river|independiente)",
    r"vs\.\s.*(en vivo|en directo)",
    r"(apertura|clausura).*en vivo",
    r"\b\d+\s*-\s*\d+\b.*\b(gol|penales?|apertura|clausura|fecha\s+\d)\b",
    r"\b(primer|segundo)\s+tiempo\b.*\b(gol|lesion|messi|partido|empat|argen|colom|uruguay|penal)",
    r"\b(gol|lesion|messi|partido|empat)\b.*\b(primer|segundo)\s+tiempo\b",
    r"\bse juega\b.*\b(primer|segundo)\b",
    r"\bdebut[oó]?\b.*\bprimera\b",
    r"\bgol de\b",
    r"\bcampe[oó]n\b",
    r"\bpenales\b.*\bganó\b|\bganó\b.*\bpenales\b",
    r"\bcopa libertadores\b",
    r"\bcopa am[eé]rica\b",
    r"\bcopa sudamericana\b",
    r"\bsupercopa\b",
    r"\bmundial\s+\d{4}\b",
    r"\b(pe[ñn]arol|nacional|cerro porte[ñn]o|olimpia|libertad)\b.*(gan[oó]|perdi[oó]|empat[oó]|venci[oó])",
    r"(gan[oó]|perdi[oó]|empat[oó]|venci[oó]).*\b(pe[ñn]arol|nacional|cerro porte[ñn]o|olimpia|libertad)\b",
    r"\b\d+\s*fotos?\b",
    r"\btobillo\b.*\bmessi\b|\bmessi\b.*\btobillo\b",
    r"\bsu[aá]rez\b.*\bdescuentos?\b",
    r"\bciclismo\b.*\bcampeonato\b|\bcampeonato\b.*\bciclismo\b",
    r"\bdeportivo maldonado\b",
    r"\bbicampe[oó]n\b",
    r"\balargue\b",
    # Lottery / quiniela / numbers games — routine roundups, not news
    r"\bquiniela\b",
    r"\bquini ?6\b",
    r"\blotería\b|\bloteria\b",
    r"\btelekino\b",
    r"\bbrinco\b",
    r"\btómbola\b|\btombola\b",
    r"\bpowerball\b",
    r"\bmega ?millions?\b",
    r"\bla primera\b.*\bnúmero\b",
    r"resultados? del sorteo",
    r"sorteo de hoy",
    r"números? ganadores?",
    r"numeros? ganadores?",
]

def _clean_scraped_title(title: str) -> str:
    """Strip section-header prefixes and newlines from scraped titles."""
    if not title or "\n" not in title:
        return (title or "").strip()
    lines = title.split("\n")
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Skip pure ALL-CAPS lines (section headers like VIDEO, FECHA 11, INTERNACIONALES)
        if re.match(r'^[A-ZÁÉÍÓÚÑ0-9 /\-]{3,}$', s) and len(s) < 60:
            continue
        cleaned.append(s)
    result = " ".join(cleaned)
    result = re.sub(r'\s+', ' ', result).strip()
    if result.startswith(">"):
        result = result.lstrip("> ").strip()
    return result


# ── Ranking & merge tuning ─────────────────────────────────────────────
MERGE_TITLE_SIM = 0.42        # cosine sim between cluster titles to trigger merge
MERGE_KW_JACCARD = 0.45       # Jaccard sim between cluster keyword sets
OVERVIEW_DOMESTIC_RATIO = 0.5  # min domestic fraction for overview selection
OVERVIEW_COUNTRY_RATIO = 0.3   # min country fraction for overview selection
RANK_DOMESTIC_BONUS = 2.0
RANK_CONCENTRATION_BONUS = 1.5
RANK_INTL_PENALTY = 2.0
MIN_DOMESTIC_SLOTS = 8         # of 12 Key Stories slots reserved for domestic (legacy)
TOTAL_STORY_SLOTS = 12         # legacy global cap (no longer used)
PER_TAB_DISPLAY_CAP = 10       # max Key Stories shown per tab (country / regional)
PER_TAB_BRIEF_CAP = 10         # max stories per tab that get LLM Summary+Context

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
    # Generic political/institutional words that don't discriminate events
    "casa", "rosada", "gobierno", "país", "pais", "nación", "nacion",
    "presidente", "ministro", "ministra", "congreso", "senado",
    "momento", "debate", "propuesta", "proyecto", "medida", "medidas",
    "argentina", "uruguay", "paraguay",
    # Electoral vocabulary too common to discriminate between elections
    "candidatura", "candidato", "candidata", "elecciones", "electoral",
    "partido", "campaña", "campana", "voto", "votar", "presidencial",
    "oficialismo", "oposición", "oposicion",
}


def load_articles():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM articles ORDER BY published_at DESC", conn)
    conn.close()
    if df.empty:
        return df
    for col in ["subcategory", "category_confidence", "fetch_method", "section_url"]:
        if col not in df.columns:
            df[col] = None
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["published_at"] = df["published_at"].dt.tz_localize(None)
    df["date"] = df["published_at"].dt.date
    # Only keep articles scraped in the last 36 hours
    df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce")
    cutoff = datetime.now() - timedelta(hours=36)
    recent = df[df["scraped_at"] >= cutoff]
    if len(recent) > 0:
        df = recent
    cleaned_rows = []
    for row in df.to_dict("records"):
        # Clean up titles with embedded section headers / newlines from scrapers
        raw_title = row.get("title", "")
        row["title"] = _clean_scraped_title(raw_title)
        analysis = analyze_article(
            row["title"],
            row.get("url", ""),
            source=row.get("source", ""),
            fallback=row.get("category"),
            strict=row.get("fetch_method") in {"html", "playwright"},
        )
        if not analysis["accepted"]:
            continue
        row["category"] = analysis["category"]
        row["subcategory"] = analysis["subcategory"]
        row["category_confidence"] = analysis["confidence"]
        cleaned_rows.append(row)
    df = pd.DataFrame(cleaned_rows)
    if df.empty:
        return df
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
    "Si el evento es una conmemoración, aniversario o investigación de un hecho "
    "histórico, el título DEBE reflejarlo claramente (ej: 'A 50 años de…', "
    "'Investigan…', 'Conmemoran…'). No presentes hechos históricos como si "
    "fueran noticias actuales. "
    "Solo respondé con el título, nada más."
)

_TITLE_FROM_SUMMARY_PROMPT = (
    "You are a diplomatic briefing editor. Below is a factual summary of a "
    "news event. Write ONE short, specific headline in Spanish (max 100 "
    "characters) that describes the CONCRETE event in the summary — not a "
    "generic label. Include the main actor(s) and the key fact. Do NOT use "
    "quotes, do NOT use 'EN VIVO', do NOT use vague phrasing like "
    "'y todas sus medidas' or 'lo último'. "
    "If the event is a commemoration, anniversary, or investigation of a "
    "historical fact, the headline MUST make this clear (e.g. 'A 50 años "
    "de…', 'Investigan…'). Never present historical events as current news. "
    "Reply with ONLY the headline text, nothing else."
)

_BANNED_PHRASES = (
    "Do NOT editorialize, do NOT interpret, do NOT use phrases like "
    "'highlights the importance', 'raises concerns', 'has the potential', "
    "'is relevant because', 'puts to the test', 'reflects', 'underscores', "
    "'could lead to', 'may impact', 'is likely to'. "
)

_BRIEF_PROMPT = (
    "You are a wire-service news writer producing structured briefing entries. "
    "From the following Spanish-language headlines covering the same event, "
    "produce a JSON object with TWO fields:\n\n"
    '- "summary": ONE short paragraph IN ENGLISH (3 sentences, max 400 characters) '
    "sticking strictly to the facts: what happened, who, where, when. "
    + _BANNED_PHRASES +
    "Do not use quotes, do not invent facts not present in the headlines.\n\n"
    '- "context": ONE single sentence IN ENGLISH (max 180 characters) providing factual '
    "background — what ongoing process, policy, institution, or longstanding situation "
    "this event belongs to. EXPLANATORY, NOT predictive. State background facts only "
    "(e.g., 'Subsidy reform has been a central axis of the government's fiscal "
    "adjustment since December 2023.'). Do NOT speculate about consequences. "
    'If you cannot provide neutral background, return an empty string for "context".\n\n'
    "Reply with ONLY the JSON object. No markdown fences, no preamble, no commentary."
)

_BRIEF_WITH_BODY_PROMPT = (
    "You are a wire-service news writer producing structured briefing entries. "
    "From the following Spanish-language headlines AND article excerpts covering "
    "the same event, produce a JSON object with TWO fields:\n\n"
    '- "summary": ONE short paragraph IN ENGLISH (3 sentences, max 400 characters) '
    "sticking strictly to the facts: what happened, who, where, when. "
    "Use the excerpts to add specific details (names, numbers, dates) not in the "
    "headlines alone. "
    + _BANNED_PHRASES +
    "Do not use quotes, do not invent facts not present in the sources.\n\n"
    '- "context": ONE single sentence IN ENGLISH (max 180 characters) providing factual '
    "background — what ongoing process, policy, institution, or longstanding situation "
    "this event belongs to. EXPLANATORY, NOT predictive. State background facts only "
    "(e.g., 'Subsidy reform has been a central axis of the government's fiscal "
    "adjustment since December 2023.'). Do NOT speculate about consequences. "
    'If you cannot provide neutral background, return an empty string for "context".\n\n'
    "Reply with ONLY the JSON object. No markdown fences, no preamble, no commentary."
)


# QeQ roster: ~1,500 Cono Sur public figures from curated Excel files under QeQ/.
# This is the single source of truth for the people roster — the old
# known_people.py module is superseded by the richer, more up-to-date
# QeQ Excel workbooks.
try:
    from qeq_loader import load_qeq_people
    _QEQ_PEOPLE = load_qeq_people()
except Exception:
    _QEQ_PEOPLE = []

# Roles that signal the LLM was guessing — drop these entries entirely
_GENERIC_ROLES = {
    "politician", "official", "leader", "figure", "person", "member",
    "public figure", "political figure", "government official",
    "party member", "national figure", "spokesperson", "spokesman",
}
# _PERSON_CONNECTORS and _NON_PERSON_NAME_TOKENS are imported from name_utils


# _normalize_person_name and _looks_like_person_name are imported from name_utils


def _count_name_mentions(name, titles):
    """Count conservative title mentions for a person-like name."""
    norm_name = _normalize_person_name(name)
    if not norm_name:
        return 0
    parts = norm_name.split()
    if len(parts) < 2:
        return 0
    last_name = parts[-1]
    count = 0
    for title in titles:
        norm_title = _normalize_person_name(title)
        tokens = set(norm_title.split())
        if norm_name in norm_title:
            count += 1
            continue
        if len(last_name) >= 5 and last_name not in _AMBIGUOUS_SHORT_LASTNAMES and last_name in tokens:
            count += 1
    return count


def _merged_known_people():
    """Return the QeQ roster as the single source of truth.

    Deduplicates by normalized name so there's at most one entry per person.
    """
    seen = set()
    merged = []
    for entry in _QEQ_PEOPLE:
        norm = _normalize_person_name(entry.get("name", ""))
        if not norm or norm in seen:
            continue
        seen.add(norm)
        merged.append(entry)
    return merged


_ALL_KNOWN_PEOPLE = _merged_known_people()


def _build_known_people_index():
    """Build lookup indices for identifying people in news titles.

    Returns three dicts:
      * full_idx     — normalized full name → entry
      * fl_idx       — "firstword lastword" → entry (fallback for long names)
      * alias_idx    — normalized alias → entry (QeQ nicknames)
    """
    full_idx = {}
    fl_idx = {}
    alias_idx = {}
    for entry in _ALL_KNOWN_PEOPLE:
        norm = _normalize_person_name(entry.get("name", ""))
        if not norm:
            continue
        full_idx.setdefault(norm, entry)
        words = norm.split()
        if len(words) >= 2:
            fl_idx.setdefault(f"{words[0]} {words[-1]}", entry)
        for alias in entry.get("aliases", []) or []:
            nalias = _normalize_person_name(alias)
            # Require ≥2 tokens or ≥6 chars to avoid matching common words
            if not nalias:
                continue
            if len(nalias.split()) < 2 and len(nalias) < 5:
                continue
            alias_idx.setdefault(nalias, entry)
    return full_idx, fl_idx, alias_idx


_KNOWN_FULL_IDX, _KNOWN_FL_IDX, _KNOWN_ALIAS_IDX = _build_known_people_index()


def _match_known_person(name):
    """Return canonical known-person entry for a name, or None."""
    norm = _normalize_person_name(name)
    if not norm:
        return None
    if norm in _KNOWN_FULL_IDX:
        return _KNOWN_FULL_IDX[norm]
    if norm in _KNOWN_ALIAS_IDX:
        return _KNOWN_ALIAS_IDX[norm]
    words = norm.split()
    if len(words) >= 2:
        fl_key = f"{words[0]} {words[-1]}"
        if fl_key in _KNOWN_FL_IDX:
            return _KNOWN_FL_IDX[fl_key]
    return None


def _entry_source(entry):
    return entry.get("source") or "qeq"


# Prominence tiers used to pick the canonical entry per last name. Higher
# score = more likely to be the person a naked "Riera" / "Paredes" / "López"
# in a Cono Sur headline is referring to.
_ROLE_PROMINENCE = [
    (re.compile(r"\bpresident", re.I), 100),
    (re.compile(r"vice ?president", re.I), 95),
    (re.compile(r"jefe de gabinete|chief of the cabinet|chief of staff", re.I), 92),
    (re.compile(r"\bministr[oa]\b|\bminister\b|\bsecretari[oa] general\b", re.I), 90),
    (re.compile(r"gobernador|governor|intendent[ae]", re.I), 80),
    (re.compile(r"\bsenador[a]?\b|\bsenator\b", re.I), 70),
    (re.compile(r"diputad[oa]|deputy|legislador", re.I), 60),
    (re.compile(r"juez|fiscal|supreme court|corte suprema|justice", re.I), 58),
    (re.compile(r"embajador|ambassador", re.I), 55),
    (re.compile(r"l[ií]der|leader|party", re.I), 50),
    (re.compile(r"ceo|founder|president.*company|empresari[oa]|directiv[oa]", re.I), 45),
    (re.compile(r"ex ?president", re.I), 40),
    (re.compile(r"futbolist|deportist|tenist|basquetbol|player", re.I), 20),
    (re.compile(r"narco|crimen|criminal|trafican", re.I), 5),
]


def _role_prominence(entry) -> int:
    role = (entry.get("role") or "").lower()
    for pat, score in _ROLE_PROMINENCE:
        if pat.search(role):
            return score
    return 30  # default mid-tier


def _build_lastname_prominence_map():
    """Pick the single canonical entry for each last name in the roster.

    When two people share a last name (e.g. Enrique Riera, Minister of the
    Interior; Fausto Riera, narco), we want naked-surname matches in
    headlines to land on the more prominent figure. This builds::

        {last_name: (canonical_entry, is_ambiguous)}

    where ``is_ambiguous`` is True when more than one entry shares that last
    name. Even ambiguous last names get a canonical pick — the matcher then
    decides whether to trust it based on how dominant the canonical entry is.
    """
    buckets: dict[str, list[dict]] = {}
    for entry in _ALL_KNOWN_PEOPLE:
        norm = _normalize_person_name(entry.get("name", ""))
        words = norm.split()
        if len(words) < 2:
            continue
        buckets.setdefault(words[-1], []).append(entry)

    result: dict[str, tuple[dict, bool]] = {}
    for last_name, entries in buckets.items():
        if len(entries) == 1:
            result[last_name] = (entries[0], False)
            continue
        # Sort by prominence desc, then by source (KP first), then by role length
        entries.sort(
            key=lambda e: (
                -_role_prominence(e),
                0 if _entry_source(e) == "kp" else 1,
                -len(e.get("role", "")),
            )
        )
        top = entries[0]
        second = entries[1]
        # Keep the canonical pick if it clearly outranks the runner-up.
        # Presidents/VPs (≥95) need only a 5-point gap; ministers/governors
        # (≥80) need 10; others need 20.
        top_prom = _role_prominence(top)
        gap = top_prom - _role_prominence(second)
        if top_prom >= 95:
            min_gap = 5
        elif top_prom >= 80:
            min_gap = 10
        else:
            min_gap = 20
        if gap >= min_gap:
            result[last_name] = (top, True)
        else:
            result[last_name] = (top, True)
    return result


_LASTNAME_CANONICAL = _build_lastname_prominence_map()

# Short last names that would collide with ordinary Spanish words if matched.
# These are excluded from the surname scan even when canonical.
_AMBIGUOUS_SHORT_LASTNAMES = {
    "cruz", "rosa", "mano", "caso", "plan", "hora", "mesa", "gato",
    "vaca", "gana", "para", "mejor", "solo", "cosa",
    "guerra", "real", "paz", "campo", "rico", "blanco",
    "fuerte", "franco", "pastor", "moral", "bueno", "leal",
}
# NOTE: "negro" removed — it blocked Uruguay's Interior Minister Carlos Negro.
# The prominence system (80+ = ministro) now handles disambiguation correctly.
# NOTE: "pena" removed — it blocked surname matching for Santiago Peña
# (president of Paraguay). The prominence system + canonical map now
# handle disambiguation (Peña has prominence 100, far above any runner-up).
# "guerra" added — "guerra" (war) appears constantly in international
# headlines and would false-match to singers/athletes named Guerra.


# Some figures are best known by a distinctive FIRST name (e.g. Nicanor,
# Yamandú, Lilita) more than by their last name. We allow first-name matching
# when the first name is (a) reasonably long and (b) unique to a single
# prominent entry in the roster.
_COMMON_FIRST_NAMES = {
    "juan", "jose", "maria", "luis", "carlos", "jorge", "pedro", "pablo",
    "ana", "laura", "silvia", "marcelo", "fernando", "roberto", "ricardo",
    "diego", "daniel", "javier", "martin", "eduardo", "alberto", "andres",
    "miguel", "rafael", "gabriel", "victor", "oscar", "hugo", "raul",
    "gustavo", "alejandro", "rodrigo", "cristian", "sergio", "manuel",
    "hector", "antonio", "francisco",
}

# Spanish words that happen to be first names — too ambiguous for first-name-
# only matching because they appear constantly in headlines with their
# ordinary word meaning (e.g. "libertad" = freedom, not actress Libertad Lamarque).
_SPANISH_WORD_FIRST_NAMES = {
    "libertad", "esperanza", "victoria", "pilar", "dolores", "mercedes",
    "rosario", "concepcion", "trinidad", "soledad", "consuelo", "socorro",
    "asuncion", "amparo", "milagros", "remedios", "argentina",
}


def _build_firstname_canonical_map():
    buckets: dict[str, list[dict]] = {}
    for entry in _ALL_KNOWN_PEOPLE:
        norm = _normalize_person_name(entry.get("name", ""))
        words = norm.split()
        if len(words) < 2:
            continue
        first = words[0]
        if len(first) < 6 or first in _COMMON_FIRST_NAMES or first in _SPANISH_WORD_FIRST_NAMES:
            continue
        buckets.setdefault(first, []).append(entry)

    result: dict[str, dict] = {}
    for first, entries in buckets.items():
        if len(entries) == 1:
            result[first] = entries[0]
            continue
        entries.sort(
            key=lambda e: (
                -_role_prominence(e),
                -len(e.get("role", "")),
            )
        )
        # Require a clear prominence gap to trust a first-name-only match
        if _role_prominence(entries[0]) - _role_prominence(entries[1]) >= 30:
            result[first] = entries[0]
    return result


_FIRSTNAME_CANONICAL = _build_firstname_canonical_map()


_KNOWN_ORGANIZATIONS = [
    {"name": "La Libertad Avanza", "country": "argentina", "role": "Political party", "bio": "Ruling libertarian party led by Javier Milei.", "aliases": ["LLA"]},
    {"name": "PRO", "country": "argentina", "role": "Political party", "bio": "Center-right Argentine party founded by Mauricio Macri.", "aliases": ["Propuesta Republicana"]},
    {"name": "Union por la Patria", "country": "argentina", "role": "Peronist coalition", "bio": "Main Peronist electoral coalition in national politics.", "aliases": ["UxP"]},
    {"name": "UCR", "country": "argentina", "role": "Political party", "bio": "Historic Radical Civic Union party in Argentina.", "aliases": ["Union Civica Radical"]},
    {"name": "FMI", "country": "international", "role": "International financial institution", "bio": "International Monetary Fund, central to regional fiscal negotiations.", "aliases": ["Fondo Monetario Internacional"]},
    {"name": "BCRA", "country": "argentina", "role": "Central bank", "bio": "Central Bank of the Argentine Republic.", "aliases": ["Banco Central"]},
    {"name": "Frente Amplio", "country": "uruguay", "role": "Political coalition", "bio": "Uruguayan left-wing coalition currently in government.", "aliases": []},
    {"name": "Partido Nacional", "country": "uruguay", "role": "Political party", "bio": "Traditional Uruguayan center-right party.", "aliases": ["Blanco", "Blancos"]},
    {"name": "Partido Colorado", "country": "paraguay", "role": "Political party", "bio": "Colorado Party, Paraguay's dominant political force.", "aliases": ["ANR", "Asociacion Nacional Republicana"]},
    {"name": "PLRA", "country": "paraguay", "role": "Political party", "bio": "Partido Liberal Radical Autentico, main opposition party in Paraguay.", "aliases": ["Partido Liberal Radical Autentico"]},
    {"name": "IPS", "country": "paraguay", "role": "Social security institution", "bio": "Instituto de Prevision Social, Paraguay's main social security and health system.", "aliases": []},
    {"name": "ANDE", "country": "paraguay", "role": "State electricity company", "bio": "National Electricity Administration of Paraguay.", "aliases": []},
    {"name": "MOPC", "country": "paraguay", "role": "Public works ministry", "bio": "Ministry of Public Works and Communications of Paraguay.", "aliases": []},
    {"name": "MEF", "country": "paraguay", "role": "Economy ministry", "bio": "Ministry of Economy and Finance of Paraguay.", "aliases": []},
    {"name": "Petropar", "country": "paraguay", "role": "State fuel company", "bio": "Paraguay's state oil and fuel company.", "aliases": []},
    {"name": "Mercosur", "country": "regional", "role": "Regional bloc", "bio": "South American trade bloc linking Argentina, Brazil, Paraguay and Uruguay.", "aliases": []},
    {"name": "PIT-CNT", "country": "uruguay", "role": "Trade union federation", "bio": "Uruguay's main labor federation.", "aliases": []},
]


def _entity_confidence(mentions, basis):
    base = {
        "full-name": 0.95,
        "full-name-llm": 0.7,
        "multi-alias": 0.9,
        "single-alias": 0.82,
        "last-name": 0.78,
        "first-name": 0.74,
        "org-name": 0.9,
        "org-alias": 0.82,
    }.get(basis, 0.65)
    return round(min(0.99, base + min(mentions - 2, 4) * 0.03), 2)


def _annotate_entity(entry, mentions, basis, entity_type):
    out = dict(entry)
    out["mentions"] = mentions
    out["confidence"] = _entity_confidence(mentions, basis)
    out["match_basis"] = basis
    out["entity_type"] = entity_type
    return out


def _scan_titles_for_known_people(titles):
    """Return high-confidence people detected across the title corpus."""
    if not titles or not _ALL_KNOWN_PEOPLE:
        return []
    norm_titles = [_normalize_person_name(t) for t in titles]
    title_tokens = [set(nt.split()) for nt in norm_titles]
    found = {}
    for entry in _ALL_KNOWN_PEOPLE:
        norm = _normalize_person_name(entry.get("name", ""))
        if not norm:
            continue
        words = norm.split()
        last_name = words[-1] if words else ""
        canonical = _LASTNAME_CANONICAL.get(last_name)
        allow_last = (
            len(last_name) >= 4
            and last_name not in _AMBIGUOUS_SHORT_LASTNAMES
            and canonical is not None
            and canonical[0] is entry
        )
        first_name = words[0] if words else ""
        allow_first = (
            len(first_name) >= 6
            and _FIRSTNAME_CANONICAL.get(first_name) is entry
        )
        multi_word_aliases = []
        single_word_aliases = []
        for alias in entry.get("aliases", []) or []:
            nalias = _normalize_person_name(alias)
            if not nalias:
                continue
            if len(nalias.split()) >= 2:
                multi_word_aliases.append(nalias)
            elif len(nalias) >= 5:
                single_word_aliases.append(nalias)
        count = 0
        basis = None
        for nt, toks in zip(norm_titles, title_tokens):
            if not nt:
                continue
            if norm in nt:
                count += 1
                basis = basis or "full-name"
                continue
            if any(na in nt for na in multi_word_aliases):
                count += 1
                basis = basis or "multi-alias"
                continue
            if any(sa in toks for sa in single_word_aliases):
                count += 1
                basis = basis or "single-alias"
                continue
            if allow_last and last_name in toks:
                count += 1
                basis = basis or "last-name"
                continue
            if allow_first and first_name in toks:
                count += 1
                basis = basis or "first-name"
        # High-prominence figures (presidents, ministers, governors: ≥80)
        # are accepted with a single mention; others still require ≥2.
        min_count = 1 if _role_prominence(entry) >= 80 else 2
        if count >= min_count and basis:
            found[norm] = _annotate_entity(entry, count, basis, "person")
    ranked = sorted(
        found.values(),
        key=lambda e: (-e["mentions"], -e["confidence"], e["name"]),
    )
    return ranked[:30]


def _scan_titles_for_known_organizations(titles):
    """Return frequently mentioned organizations, parties, and institutions."""
    if not titles:
        return []
    norm_titles = [_normalize_person_name(t) for t in titles]
    title_tokens = [set(nt.split()) for nt in norm_titles]
    found = {}
    for entry in _KNOWN_ORGANIZATIONS:
        norm = _normalize_person_name(entry["name"])
        aliases = [_normalize_person_name(a) for a in entry.get("aliases", []) if a]
        count = 0
        basis = None
        for nt, toks in zip(norm_titles, title_tokens):
            full_match = False
            if norm:
                if len(norm.split()) == 1 and len(norm) <= 4:
                    full_match = norm in toks
                else:
                    full_match = norm in nt
            if full_match:
                count += 1
                basis = basis or "org-name"
                continue
            matched_alias = False
            for alias in aliases:
                if len(alias.split()) >= 2 and alias in nt:
                    matched_alias = True
                    basis = basis or "org-alias"
                    break
                if len(alias.split()) == 1 and alias in toks:
                    matched_alias = True
                    basis = basis or "org-alias"
                    break
            if matched_alias:
                count += 1
        if count >= 2:
            found[norm] = _annotate_entity(entry, count, basis or "org-name", "organization")
    ranked = sorted(
        found.values(),
        key=lambda e: (-e["mentions"], -e["confidence"], e["name"]),
    )
    return ranked[:12]


_GLOSSARY_PROMPT = (
    "You are a regional news analyst. From the following Spanish-language news "
    "headlines from Argentina, Uruguay, and Paraguay, identify the most prominent "
    "people mentioned (politicians, officials, judges, business leaders, etc.).\n\n"
    'Return a JSON object with a single field "people" containing an array of up '
    "to 12 entries. Each entry must have:\n"
    '- "name": full name as commonly written\n'
    '- "role": their official position or known public role\n'
    '- "country": one of "argentina", "uruguay", "paraguay", "regional", "international"\n'
    '- "bio": ONE single sentence IN ENGLISH (max 140 characters) giving factual '
    "background — party, institution, what they are known for. Explanatory only. "
    "NO opinions, NO description of recent events, NO predictions.\n\n"
    "Rules:\n"
    "- Skip anyone mentioned only once across the headlines.\n"
    "- Order by prominence (most mentioned first).\n"
    "- If you do not know a person's role with confidence, omit them. Do NOT guess.\n"
    "- Return ONLY the JSON object. No markdown fences, no preamble."
)


def _llm_extract_glossary(titles):
    """Extract a list of prominent people from headlines.

    Strategy:
      1. Auto-seed: scan titles for KNOWN_PEOPLE names (≥2 occurrences). These
         are 100% reliable — canonical role + bio from known_people.py.
      2. Call LLM (Gemini first, Groq fallback) to discover other figures.
      3. Override LLM-returned entries that match KNOWN_PEOPLE.
      4. Drop entries with generic/low-confidence roles.
      5. Merge auto-seeded + LLM, dedupe, cap at 12.
    """
    if not titles:
        return []

    # ── Step 1: auto-seed from KNOWN_PEOPLE present in the corpus ──
    seed_entries = _scan_titles_for_known_people(titles)
    seed_normalized = {_normalize_person_name(e["name"]) for e in seed_entries}

    # ── Step 2: LLM extraction (uses unified cascade) ──
    bullet_list = "\n".join(f"- {t}" for t in titles)
    user_msg = f"Headlines:\n{bullet_list}"
    raw_text = _llm_chat_cascade(_GLOSSARY_PROMPT, user_msg, json_mode=True, max_tokens=1500)

    llm_entries = []
    if raw_text:
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        try:
            obj = json.loads(text)
            llm_entries = obj.get("people", []) or []
        except Exception:
            llm_entries = []

    # ── Step 3+4: clean LLM entries, override with KNOWN, drop generic roles ──
    valid_countries = {"argentina", "uruguay", "paraguay", "regional", "international"}
    cleaned_llm = []
    seen_names = set(seed_normalized)
    for e in llm_entries:
        if not isinstance(e, dict):
            continue
        name = (e.get("name") or "").strip()
        role = (e.get("role") or "").strip()
        country = (e.get("country") or "").strip().lower()
        bio = (e.get("bio") or "").strip()
        if not name or not role or country not in valid_countries:
            continue
        if not _looks_like_person_name(name):
            continue

        # Override with canonical KNOWN_PEOPLE entry if matched
        match = _match_known_person(name)
        if match:
            norm = _normalize_person_name(match["name"])
            if norm in seen_names:
                continue  # already seeded
            seen_names.add(norm)
            mentions = _count_name_mentions(match["name"], titles)
            cleaned_llm.append(_annotate_entity(match, mentions, "full-name-llm", "person"))
            continue

        # Drop generic-role entries (signal of LLM guessing)
        if role.lower().strip() in _GENERIC_ROLES:
            continue
        if _count_name_mentions(name, titles) < 2:
            continue

        norm = _normalize_person_name(name)
        if norm in seen_names:
            continue
        seen_names.add(norm)

        if len(bio) > 200:
            bio = bio[:197] + "..."
        mentions = _count_name_mentions(name, titles)
        cleaned_llm.append({
            "name": name,
            "role": role,
            "country": country,
            "bio": bio,
            "mentions": mentions,
            "confidence": _entity_confidence(mentions, "full-name-llm"),
            "match_basis": "full-name-llm",
            "entity_type": "person",
        })

    # ── Step 5: merge — seeded first (highest confidence), then LLM extras ──
    merged = list(seed_entries) + cleaned_llm
    return merged[:12]


def _llm_chat_cascade(system_prompt, user_msg, json_mode=True, max_tokens=500):
    """Try LLM providers in order: Mistral → Cerebras → Groq → Gemini.

    Returns the raw text response, or None if all fail.
    """
    # ── Mistral ──
    if MISTRAL_API_KEY:
        try:
            body = {
                "model": "mistral-small-latest",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            resp = http_requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
                json=body,
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    # ── Cerebras ──
    if CEREBRAS_API_KEY:
        try:
            body = {
                "model": "llama3.1-8b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            resp = http_requests.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
                json=body,
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    # ── Groq ──
    if GROQ_API_KEY:
        for attempt in range(2):
            try:
                body = {
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                }
                if json_mode:
                    body["response_format"] = {"type": "json_object"}
                resp = http_requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json=body,
                    timeout=15,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception:
                if attempt == 0:
                    time.sleep(2)

    # ── Gemini (fallback) ──
    if GEMINI_API_KEY:
        try:
            gen_config = {"temperature": 0.3, "maxOutputTokens": max_tokens}
            if json_mode:
                gen_config["responseMimeType"] = "application/json"
            resp = http_requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_msg}"}]}],
                    "generationConfig": gen_config,
                },
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass

    return None


# Post-LLM filter: strip editorial phrases that small models ignore
_EDITORIAL_RE = re.compile(
    r"\b(raises concerns?|highlights? the importance|has the potential|"
    r"is relevant because|puts? to the test|reflects?|underscores?|"
    r"could lead to|may impact|is likely to|sends? a (?:clear |strong )?(?:signal|message))\b",
    re.IGNORECASE,
)


def _parse_brief_json(text):
    """Parse LLM response into (summary, context) tuple. Returns (None, None) on failure."""
    if not text:
        return None, None
    text = text.strip().strip('"\'""«»\n ')
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Try to locate JSON object boundaries if there's surrounding noise
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        obj = json.loads(text)
    except Exception:
        return None, None
    summary = (obj.get("summary") or "").strip().strip('"\'""«»')
    context = (obj.get("context") or "").strip().strip('"\'""«»')

    # Strip editorial phrases the LLM may have added despite instructions
    summary = _EDITORIAL_RE.sub("", summary).strip()
    summary = re.sub(r"\s{2,}", " ", summary)
    context = _EDITORIAL_RE.sub("", context).strip()
    context = re.sub(r"\s{2,}", " ", context)

    if not (50 < len(summary) < 450):
        return None, None
    if context and len(context) > 200:
        context = ""
    return summary, context


def _llm_synthesize_title(headlines):
    """Call LLM API to synthesize a title. Tries Mistral → Cerebras → Groq → Gemini."""
    bullet_list = "\n".join(f"- {h}" for h in headlines)
    user_msg = f"Titulares:\n{bullet_list}"
    text = _llm_chat_cascade(_TITLE_PROMPT, user_msg, json_mode=False, max_tokens=80)
    if text:
        text = text.strip('"\'""«»\n .')
        if 5 < len(text) < 130:
            return text
    return None


def _llm_synthesize_title_from_summary(summary_en):
    """Generate a specific Spanish headline derived from the English summary.

    Tries Mistral → Cerebras → Groq → Gemini.
    """
    if not summary_en or len(summary_en) < 30:
        return None
    user_msg = f"Summary:\n{summary_en}"
    text = _llm_chat_cascade(_TITLE_FROM_SUMMARY_PROMPT, user_msg, json_mode=False, max_tokens=80)
    if text:
        text = text.strip('"\'""«»\n .')
        if 5 < len(text) < 130:
            return text
    return None


def _llm_synthesize_brief(headlines):
    """Call LLM to synthesize {summary, context}. Returns (summary, context) tuple.

    Tries Mistral → Cerebras → Groq → Gemini. Returns (None, None) on failure.
    """
    bullet_list = "\n".join(f"- {h}" for h in headlines)
    user_msg = f"Titulares:\n{bullet_list}"
    text = _llm_chat_cascade(_BRIEF_PROMPT, user_msg)
    if text:
        summary, context = _parse_brief_json(text)
        if summary:
            return summary, context
    return None, None


def _llm_synthesize_brief_with_body(headlines, bodies):
    """Synthesize {summary, context} using headlines + article body excerpts.

    Tries Mistral → Cerebras → Groq → Gemini. Falls back to headline-only brief.
    """
    bullet_list = "\n".join(f"- {h}" for h in headlines)
    body_text = "\n\n".join(bodies[:3])  # max 3 bodies to fit context
    user_msg = f"Titulares:\n{bullet_list}\n\nExcerpts:\n{body_text}"
    text = _llm_chat_cascade(_BRIEF_WITH_BODY_PROMPT, user_msg)
    if text:
        summary, context = _parse_brief_json(text)
        if summary:
            return summary, context
    # Fall back to headline-only brief
    return _llm_synthesize_brief(headlines)


def _generate_story_title(articles, use_llm=False):
    """Generate a synthesized story title.

    When use_llm=True, tries LLM synthesis first.
    Otherwise (default during clustering), uses fast TF-IDF centroid selection.
    """
    if len(articles) == 1:
        return articles[0].title

    titles = list(dict.fromkeys(art.title for art in articles))  # dedupe preserving order

    # LLM synthesis only when explicitly requested (display-time, not clustering)
    if use_llm:
        llm_title = _llm_synthesize_title(titles[:6])
        if llm_title:
            return llm_title

    # Fast fallback: TF-IDF centroid-based selection
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


def _fetch_article_body(url):
    """Extract the first ~500 chars of article body text using Playwright."""
    try:
        from scraper import _get_playwright_browser
        browser = _get_playwright_browser()
        if not browser:
            return ""
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        page.wait_for_timeout(1500)

        # Try common article body selectors
        for sel in ["article p", "[itemprop='articleBody'] p",
                     "[class*='article-body'] p", "[class*='content'] p",
                     "[class*='nota'] p", "[class*='story'] p", "main p"]:
            elements = page.query_selector_all(sel)
            if elements:
                paragraphs = []
                for el in elements[:6]:
                    txt = el.inner_text().strip()
                    if len(txt) > 30:
                        paragraphs.append(txt)
                if paragraphs:
                    body = " ".join(paragraphs)[:800]
                    page.close()
                    return body

        page.close()
    except Exception:
        pass
    return ""


def _fetch_bodies_for_cluster(articles):
    """Fetch article bodies for all articles in a cluster."""
    bodies = []
    urls_tried = set()
    for art in articles:
        if art.url in urls_tried:
            continue
        urls_tried.add(art.url)
        body = _fetch_article_body(art.url)
        if body:
            bodies.append(f"[{art.source}] {body}")
    return bodies


def _generate_story_brief(articles):
    """Return (summary, context) tuple for a cluster of articles."""
    titles = list(dict.fromkeys(art.title for art in articles))
    if not titles:
        return None, None

    bodies = _fetch_bodies_for_cluster(articles[:3])
    if bodies:
        return _llm_synthesize_brief_with_body(titles[:6], bodies)
    return _llm_synthesize_brief(titles[:6])



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

    # Vectorised keyword-overlap penalty: pairs sharing fewer than
    # min_kw_overlap keywords get distance forced to 1.0.
    # Convert keyword sets to a boolean matrix for fast intersection counting.
    all_kw = sorted({kw for ks in kw_sets for kw in ks})
    if all_kw:
        kw_idx = {w: i for i, w in enumerate(all_kw)}
        kw_mat = np.zeros((n, len(all_kw)), dtype=np.float32)
        for i, ks in enumerate(kw_sets):
            for w in ks:
                kw_mat[i, kw_idx[w]] = 1.0
        overlap = kw_mat @ kw_mat.T
        mask = overlap < min_kw_overlap
        np.fill_diagonal(mask, False)
        dist[mask] = 1.0

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, t=1.0 - threshold, criterion='distance')

    groups = defaultdict(list)
    for i, lbl in enumerate(labels):
        groups[lbl].append(i)
    return list(groups.values())


def _domestic_ratio(cluster):
    """Fraction of articles with category != 'internacional'."""
    if not cluster["articles"]:
        return 0.0
    domestic = sum(1 for a in cluster["articles"] if a.category != "internacional")
    return domestic / len(cluster["articles"])


def _story_rank_score(cluster):
    """Weighted score that corrects international bias.

    Domestic stories get a bonus; purely-international stories covering
    all 3 countries get a penalty so they don't crowd out local news.
    """
    base = len(cluster["sources"])
    dr = _domestic_ratio(cluster)
    domestic_bonus = dr * RANK_DOMESTIC_BONUS

    country_counts = Counter(a.country for a in cluster["articles"])
    max_share = max(country_counts.values()) / max(len(cluster["articles"]), 1)
    concentration_bonus = max_share * RANK_CONCENTRATION_BONUS

    n_countries = len(country_counts)
    intl_penalty = RANK_INTL_PENALTY if (n_countries >= 3 and dr < 0.3) else 0

    return base + domestic_bonus + concentration_bonus - intl_penalty


def _cluster_entity_country(cluster):
    """Infer the thematic country from the known people/entities in the cluster.

    Scans cluster titles against the roster and counts how many matched
    entities belong to each country. Returns (dominant_country, ratio, n_entities)
    or (None, 0, 0) if no entities are found.

    Only high-confidence entities (confidence >= 0.78, i.e. at least full-name
    or multi-alias match) are considered to reduce false positives.
    """
    titles = [a.title for a in cluster["articles"]]
    people = _scan_titles_for_known_people(titles)
    if not people:
        return None, 0.0, 0
    country_counts = Counter()
    for p in people:
        # Only trust high-confidence matches for country inference
        if p.get("confidence", 0) < 0.78:
            continue
        c = p.get("country", "").lower()
        if c in ("argentina", "uruguay", "paraguay"):
            country_counts[c] += 1
    total = sum(country_counts.values())
    if not country_counts or total == 0:
        return None, 0.0, 0
    dominant, count = country_counts.most_common(1)[0]
    return dominant, count / total, total


_FOREIGN_COUNTRY_PATTERNS = re.compile(
    r"\b(?:estados unidos|ee\.?\s*uu\.?|trump|biden|harris|kamala|washington|"
    r"china|rusia|ucrania|iran|irak|siria|gaza|israel|palestina|libano|"
    r"brasil|m[eé]xico|colombia|venezuela|per[uú]|chile|bolivia|ecuador|"
    r"espa[ñn]a|francia|alemania|italia|jap[oó]n|corea|india|turqu[ií]a|"
    r"nicaragua|panam[aá]|costa rica|cuba|honduras|somalia|pakist[aá]n|"
    r"irland[eéa]s?|hungar[ií]a|orb[aá]n|canad[aá]|australia|sud[aá]frica|"
    r"arab[ei]|emiratos|qatar|egipto|marruecos|argelia|nigeria|"
    r"reino unido|gran breta[ñn]a|inglaterra|escocia|"
    r"peruanos?|elecciones? .* peru|ch[aá]vez|"
    r"wall street|nasa|artemis|kremlin|pentagon[oe]|otan|nato|europa|"
    r"netanyahu|putin|macron|zelensky|petro|lula|maduro|bukele)\b",
    re.IGNORECASE,
)

# Cono Sur references that cancel out a foreign match (the story connects
# to the region even if foreign countries are mentioned).
_CONO_SUR_PATTERNS = re.compile(
    r"\b(?:argentin|urugua|paragua|buenos aires|montevideo|asunci[oó]n|"
    r"milei|bullrich|caputo|orsi|pe[ñn]a|mercosur|cono sur)\b",
    re.IGNORECASE,
)


def _titles_mention_foreign_country(cluster) -> bool:
    """Return True if cluster titles predominantly reference non-Cono-Sur places."""
    foreign_hits = 0
    local_hits = 0
    for a in cluster["articles"]:
        t = a.title
        if _FOREIGN_COUNTRY_PATTERNS.search(t):
            foreign_hits += 1
        if _CONO_SUR_PATTERNS.search(t):
            local_hits += 1
    # If more titles mention foreign places than local ones, it's foreign
    return foreign_hits > local_hits and foreign_hits >= 2


def _cluster_country_assignment(cluster):
    """Assign a cluster to one of: 'argentina', 'uruguay', 'paraguay',
    'regional', or 'internacional'.

    Rules (in order):
    1. If ≥70% of articles are categorised 'internacional' AND no strong
       Cono Sur entity signal → 'internacional'.
    2. Entity override: if ≥2 high-confidence Cono Sur entities detected
       AND ≥80% belong to one country, assign there (prevents an AR-media
       story about Paraguayan politics from landing in the AR tab).
    3. Multi-country domestic coverage (≥2 countries with ≥2 sources
       each) → 'regional'.
    4. Otherwise → dominant source country.
    """
    # Compute entity country once (expensive) and reuse
    entity_country, entity_ratio, n_ent = _cluster_entity_country(cluster)

    # ── Step 1: internacional check ──
    dr = _domestic_ratio(cluster)
    if dr < 0.30:
        if entity_country and n_ent >= 2 and entity_ratio >= 0.70:
            return entity_country
        return "internacional"

    # ── Step 1b: foreign-topic override ──
    if _titles_mention_foreign_country(cluster):
        has_cono_sur_entity = entity_country and entity_country in ("argentina", "uruguay", "paraguay")
        if not has_cono_sur_entity:
            return "internacional"

    # ── Step 3: source-based counts ──
    sources_by_country = defaultdict(set)
    for art in cluster["articles"]:
        sources_by_country[art.country].add(art.source)
    counts = {c: len(s) for c, s in sources_by_country.items()}

    if not counts:
        return "regional"

    # Single source-country is straightforward
    if len(counts) == 1:
        only_country = next(iter(counts))
        # Entity override only when strong evidence (≥2 entities, ≥80%)
        if (entity_country and n_ent >= 2
                and entity_ratio >= 0.80 and entity_country != only_country):
            return entity_country
        return only_country

    # Multi-country sources
    # Entity override: if ≥2 high-confidence actors and 80%+ from one
    # country, that's the thematic country.
    if entity_country and n_ent >= 2 and entity_ratio >= 0.80:
        return entity_country

    # Regional: ≥2 Cono Sur countries contribute sources AND the story
    # is genuinely about the region (not just international news covered
    # by multiple countries' media).
    cono_sur_countries = sum(1 for c in counts if c in ("argentina", "uruguay", "paraguay"))
    if cono_sur_countries >= 2 and dr >= 0.60:
        # Extra guard: check titles for foreign country references that
        # would make this an international story mis-categorized as domestic.
        if not _titles_mention_foreign_country(cluster):
            return "regional"

    # Otherwise assign to dominant source country
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _pick_country_top_story(clusters, country, df, used_titles):
    """Return (title_text, cluster_or_None) for the day's top story for a country.

    Replicates the 3-pass logic of the original build_overview but for one
    country at a time. Mutates `used_titles` to prevent reuse across countries.
    """
    # Pass 1: strict domestic — high domestic_ratio AND country_ratio
    candidates = []
    for c in clusters:
        if country not in c["countries"] or not c["multi"] or c["noise"]:
            continue
        if c["title"] in used_titles:
            continue
        country_arts = [a for a in c["articles"] if a.country == country]
        if not country_arts:
            continue
        domestic_arts = [a for a in country_arts if a.category != "internacional"]
        dr = len(domestic_arts) / len(country_arts)
        cr = len(country_arts) / max(len(c["articles"]), 1)
        if dr >= OVERVIEW_DOMESTIC_RATIO and cr >= OVERVIEW_COUNTRY_RATIO:
            candidates.append((c, dr, cr))

    if candidates:
        candidates.sort(key=lambda x: (x[1], x[2], len(x[0]["sources"]), x[0]["size"]), reverse=True)
        best = candidates[0][0]
        used_titles.add(best["title"])
        return best["title"], best

    # Pass 2: any multi-source cluster touching the country
    candidates = []
    for c in clusters:
        if country not in c["countries"] or not c["multi"] or c["noise"]:
            continue
        if c["title"] in used_titles:
            continue
        country_arts = [a for a in c["articles"] if a.country == country]
        cr = len(country_arts) / max(len(c["articles"]), 1)
        candidates.append((c, cr))

    if candidates:
        candidates.sort(key=lambda x: (x[1], len(x[0]["sources"]), x[0]["size"]), reverse=True)
        best = candidates[0][0]
        used_titles.add(best["title"])
        return best["title"], best

    # Pass 3: single-source fallback from this country, prefer non-international
    cdf = df[df["country"] == country]
    for _, row in cdf.iterrows():
        if (not _is_noise(row["title"]) and row["title"] not in used_titles
                and row["category"] != "internacional"):
            used_titles.add(row["title"])
            return row["title"], None

    for _, row in cdf.iterrows():
        if not _is_noise(row["title"]) and row["title"] not in used_titles:
            used_titles.add(row["title"])
            return row["title"], None

    return "Sin cobertura reciente.", None


def _merge_similar_clusters(clusters):
    """Post-clustering merge: combine clusters about the same event.

    Uses title cosine similarity + keyword Jaccard to detect duplicates.
    Union-find handles transitive merges.  Zero LLM calls.
    """
    if len(clusters) < 2:
        return clusters

    # Build keyword sets from ALL article titles per cluster (broader than synth title)
    cluster_kw = []
    for c in clusters:
        kw = set()
        for art in c["articles"]:
            kw |= _keywords(art.title)
        cluster_kw.append(kw)

    # TF-IDF on synthesised titles
    titles_norm = [_normalize(c["title"]) for c in clusters]
    vec = TfidfVectorizer(
        token_pattern=r"[a-záéíóúñü]{3,}",
        stop_words=list(STOPWORDS),
    )
    try:
        tfidf = vec.fit_transform(titles_norm)
        sim_matrix = cosine_similarity(tfidf)
    except ValueError:
        sim_matrix = np.zeros((len(clusters), len(clusters)))

    # Union-find
    parent = list(range(len(clusters)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            # Skip if both are noise or singletons
            if clusters[i]["noise"] and clusters[j]["noise"]:
                continue
            # Title cosine similarity
            title_sim = sim_matrix[i, j]
            # Keyword Jaccard
            ki, kj = cluster_kw[i], cluster_kw[j]
            jaccard = len(ki & kj) / max(len(ki | kj), 1)
            if title_sim >= MERGE_TITLE_SIM or jaccard >= MERGE_KW_JACCARD:
                union(i, j)

    # Group by root
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(len(clusters)):
        groups[find(i)].append(i)

    merged = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(clusters[members[0]])
            continue

        # Merge: combine articles, take title from largest sub-cluster
        all_articles = []
        seen_urls = set()
        best_idx = max(members, key=lambda m: clusters[m]["size"])
        for m in members:
            for art in clusters[m]["articles"]:
                if art.url not in seen_urls:
                    seen_urls.add(art.url)
                    all_articles.append(art)

        all_indices = set()
        for m in members:
            all_indices |= clusters[m]["indices"]

        sources = sorted(set(a.source for a in all_articles))
        countries = sorted(set(a.country for a in all_articles))
        categories = sorted(set(a.category for a in all_articles))
        subcategories = sorted(
            {
                str(getattr(a, "subcategory")).strip()
                for a in all_articles
                if pd.notna(getattr(a, "subcategory", None)) and str(getattr(a, "subcategory")).strip()
            }
        )

        noise_count = sum(1 for a in all_articles if _is_noise(a.title))
        is_noise = noise_count > len(all_articles) * 0.5

        # Regenerate title from the merged article pool instead of inheriting
        # a stale title from the largest sub-cluster.
        merged_title = _generate_story_title(all_articles)

        merged.append({
            "articles": all_articles,
            "sources": sources,
            "countries": countries,
            "categories": categories,
            "subcategories": subcategories,
            "primary_subcategory": _cluster_primary_subcategory(all_articles),
            "lead": clusters[best_idx]["lead"],
            "title": merged_title,
            "summary": None,
            "size": len(all_articles),
            "multi": len(sources) >= 2,
            "noise": is_noise,
            "indices": all_indices,
        })

    return merged


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
    PASS1_THRESHOLD = 0.25
    PASS2_THRESHOLD = 0.40
    RECLUSTER_MIN = 10   # only split really large clusters
    MIN_KW = 3

    # Drop noise articles BEFORE clustering so lottery/quiniela/horóscopo
    # headlines can't become a "3 sources" story card.
    rows = [r for r in df.itertuples() if not _is_noise(r.title)]
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
        subcategories = sorted(
            {
                str(getattr(r, "subcategory")).strip()
                for r in cluster_rows
                if pd.notna(getattr(r, "subcategory", None)) and str(getattr(r, "subcategory")).strip()
            }
        )
        noise_count = sum(1 for r in cluster_rows if _is_noise(r.title))
        is_noise = noise_count > len(cluster_rows) * 0.5

        clusters.append({
            "articles": cluster_rows,
            "sources": sources,
            "countries": countries,
            "categories": categories,
            "subcategories": subcategories,
            "primary_subcategory": _cluster_primary_subcategory(cluster_rows),
            "lead": cluster_rows[0].title,
            "title": _generate_story_title(cluster_rows),
            "summary": None,
            "context": None,
            "size": len(cluster_rows),
            "multi": len(sources) >= 2,
            "noise": is_noise,
            "indices": set(indices),
        })

    # ── Post-merge: combine clusters about the same event ──
    clusters = _merge_similar_clusters(clusters)

    # Second merge pass: now that merged clusters have regenerated titles,
    # run again to catch near-duplicates that only became visible after
    # title re-synthesis (e.g. two clusters whose original titles differed
    # but whose regenerated titles are nearly identical).
    clusters = _merge_similar_clusters(clusters)

    clusters.sort(key=lambda c: (c["multi"], _story_rank_score(c)), reverse=True)
    return clusters


# ── HTML Builders ───────────────────────────────────────────────────────

def _build_coverage_dots(n_sources):
    """Build dot indicators for source count — one filled dot per source."""
    color = "#C1121F" if n_sources >= 5 else "#d4a800" if n_sources >= 3 else "rgba(0,48,73,0.35)"
    dots = f'<span class="cov-dot" style="background:{color}"></span>' * n_sources
    return f'<span class="coverage-dots">{dots}</span>'


def _cluster_primary_subcategory(articles):
    subcats = []
    for article in articles:
        value = getattr(article, "subcategory", None)
        if isinstance(value, str) and value.strip():
            subcats.append(value)
    if not subcats:
        return None
    return Counter(subcats).most_common(1)[0][0]


def _render_story_card(c, idx, group_by_country=False):
    """Render a single story card. If group_by_country, source lines are
    grouped under per-country sub-headers (used in the Regional tab to make
    cross-border framing comparison legible)."""
    badges = ""
    for cat in c["categories"]:
        color = CATEGORY_COLORS.get(cat, "rgba(0,48,73,0.40)")
        label = CATEGORY_LABELS.get(cat, cat)
        text_color = "#003049" if color == "#d4a800" else "white"
        badges += f'<span class="badge" style="background:{color};color:{text_color}">{label}</span> '
    if c.get("primary_subcategory"):
        sub_label = SUBCATEGORY_LABELS.get(
            c["primary_subcategory"],
            c["primary_subcategory"].replace("_", " ").title(),
        )
        badges += f'<span class="sub-badge">{sub_label}</span> '

    flags = " ".join(COUNTRY_FLAGS.get(co, "") for co in c["countries"])
    coverage = _build_coverage_dots(len(c["sources"]))

    # Source lines — flat or grouped by country
    if group_by_country:
        by_country = defaultdict(list)
        seen = set()
        for art in c["articles"]:
            if art.source in seen:
                continue
            seen.add(art.source)
            by_country[art.country].append(art)
        source_lines = ""
        for country in COUNTRIES:
            if country not in by_country:
                continue
            flag = COUNTRY_FLAGS.get(country, "")
            name = COUNTRY_NAMES.get(country, country)
            source_lines += (
                f'<div class="country-group">'
                f'<div class="cg-header">{flag} {name}</div>'
            )
            for art in by_country[country]:
                source_lines += f"""
                <div class="story-variant">
                    <span class="sv-source">{_esc(art.source)}</span>
                    <a class="sv-title" href="{_esc(art.url)}" target="_blank">{_esc(art.title)}</a>
                </div>"""
            source_lines += "</div>"
    else:
        source_lines = ""
        seen = set()
        for art in c["articles"]:
            if art.source in seen:
                continue
            seen.add(art.source)
            source_lines += f"""
            <div class="story-variant">
                <span class="sv-source">{_esc(art.source)}</span>
                <a class="sv-title" href="{_esc(art.url)}" target="_blank">{_esc(art.title)}</a>
            </div>"""

    summary_html = (
        f'<div class="story-summary">{_esc(c["summary"])}</div>'
        if c.get("summary") else ""
    )
    context_html = (
        f'<div class="story-context"><span class="ctx-label">Context</span>{_esc(c["context"])}</div>'
        if c.get("context") else ""
    )

    return f"""
    <div class="story">
        <div class="story-header">
            <span class="story-num">{idx}</span>
            {badges} {flags}
            <span class="story-sources">{len(c['sources'])} {'source' if len(c['sources']) == 1 else 'sources'} {coverage}</span>
        </div>
        <div class="story-title">{_esc(c['title'])}</div>
        {summary_html}
        {context_html}
        <div class="story-body">{source_lines}</div>
    </div>"""


def build_stories_html(clusters_to_render, group_by_country=False, empty_msg=None):
    """Render a sequence of pre-filtered, pre-sorted clusters as story cards."""
    if not clusters_to_render:
        msg = empty_msg or "No multi-source stories identified in this cycle."
        return f'<p class="muted">{msg}</p>'
    return "".join(
        _render_story_card(c, i, group_by_country=group_by_country)
        for i, c in enumerate(clusters_to_render, 1)
    )


def _confidence_label(score):
    if score >= 0.9:
        return "Alta"
    if score >= 0.78:
        return "Media"
    return "Baja"


def _render_entity_groups(entries, empty_msg):
    if not entries:
        return f'<p class="muted">{empty_msg}</p>'

    country_order = ["argentina", "uruguay", "paraguay", "regional", "international"]
    country_display = {
        "argentina": ("Argentina", COUNTRY_FLAGS.get("argentina", "")),
        "uruguay": ("Uruguay", COUNTRY_FLAGS.get("uruguay", "")),
        "paraguay": ("Paraguay", COUNTRY_FLAGS.get("paraguay", "")),
        "regional": ("Regional", ""),
        "international": ("International", ""),
    }
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["country"]].append(entry)
    for country in grouped:
        grouped[country].sort(key=lambda e: (-e.get("mentions", 0), -e.get("confidence", 0), e["name"]))

    html = ""
    for country in country_order:
        if country not in grouped:
            continue
        name, flag = country_display[country]
        cards = ""
        for entry in grouped[country]:
            bio_html = f'<div class="gloss-bio">{_esc(entry["bio"])}</div>' if entry.get("bio") else ""
            mention_text = f'{entry.get("mentions", 0)} menciones'
            confidence_text = _confidence_label(entry.get("confidence", 0.0))
            match_basis = (entry.get("match_basis") or "").replace("-", " ").title()
            meta = f"""
                <div class="gloss-meta">
                    <span class="meta-pill">{mention_text}</span>
                    <span class="meta-pill">Confianza {confidence_text}</span>
                    <span class="meta-pill">{match_basis}</span>
                </div>
            """
            cards += f"""
            <div class="gloss-card">
                <div class="gloss-name">{_esc(entry['name'])}</div>
                <div class="gloss-role">{_esc(entry['role'])}</div>
                {meta}
                {bio_html}
            </div>"""
        html += f"""
        <div class="gloss-section">
            <div class="gloss-section-header">{flag} {name}</div>
            <div class="gloss-grid">{cards}</div>
        </div>"""
    return html


def build_glossary_html(people_entries, org_entries):
    """Render separate panels for people and organizations."""
    people_html = _render_entity_groups(
        people_entries,
        "No prominent people identified in this cycle.",
    )
    org_html = _render_entity_groups(
        org_entries,
        "No institutions or parties stood out across this cycle.",
    )
    return f"""
    <div class="sec-head">Who&apos;s Who</div>
    <p class="sec-desc">People appearing repeatedly across the region&apos;s press. Cards show mention count, confidence, and the matching basis used to identify them.</p>
    {people_html}

    <div class="sec-head">Institutions To Watch</div>
    <p class="sec-desc">Parties, ministries, state companies, blocs, and labor federations that appeared repeatedly in this cycle.</p>
    {org_html}
    """


def _build_stats_bar(df):
    """Build a visual stats bar showing article distribution by category and country."""
    # Category counts
    cat_counts = df["category"].value_counts()
    total = len(df)
    cat_html = ""
    for cat in ["politica", "economia", "internacional"]:
        count = cat_counts.get(cat, 0)
        pct = round(count / max(total, 1) * 100)
        color = CATEGORY_COLORS.get(cat, "rgba(0,48,73,0.40)")
        label = CATEGORY_LABELS.get(cat, cat)
        text_color = "#003049" if color == "#d4a800" else "white"
        cat_html += (
            f'<div class="stat-item">'
            f'<span class="badge" style="background:{color};color:{text_color}">{label}</span>'
            f'<div class="stat-bar"><div class="stat-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<span class="stat-num">{count}</span>'
            f'</div>'
        )

    # Country counts
    country_counts = df["country"].value_counts()
    country_html = ""
    for country in COUNTRIES:
        count = country_counts.get(country, 0)
        pct = round(count / max(total, 1) * 100)
        flag = COUNTRY_FLAGS[country]
        country_html += (
            f'<div class="stat-item">'
            f'{flag}'
            f'<div class="stat-bar"><div class="stat-fill" style="width:{pct}%;background:#003049"></div></div>'
            f'<span class="stat-num">{count}</span>'
            f'</div>'
        )

    return f"""
    <div class="stats-row">
        <div class="stats-group">
            <div class="stats-label">By Category</div>
            {cat_html}
        </div>
        <div class="stats-group">
            <div class="stats-label">By Country</div>
            {country_html}
        </div>
    </div>"""


def build_also_reported(df, clusters, country, limit=10):
    """Headlines NOT part of any multi-source cluster — 'also reported' items.

    Round-robins across sources so no single outlet dominates the list.
    """
    # Collect indices used in displayed clusters (multi-source + promoted singles)
    used = set()
    for c in clusters:
        if c["multi"] or c.get("country"):
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
        color = CATEGORY_COLORS.get(cat, "rgba(0,48,73,0.40)")
        label = CATEGORY_LABELS.get(cat, cat)
        text_color = "#003049" if color == "#d4a800" else "white"
        date_str = row["published_at"].strftime("%d/%m") if pd.notna(row["published_at"]) else ""
        rows_html += f"""
        <div class="also-item">
            <span class="also-date">{date_str}</span>
            <span class="badge" style="background:{color};color:{text_color}">{label}</span>
            <span class="also-source">{_esc(row['source'])}</span>
            <a class="also-title" href="{_esc(row['url'])}" target="_blank">{_esc(row['title'])}</a>
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
    if df.empty:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(
                "<!DOCTYPE html><html><head><meta charset='UTF-8'>"
                "<title>Media Monitor — Error</title></head><body>"
                "<h1>No articles found</h1>"
                "<p>The database is empty. Run the scraper first: "
                "<code>python run.py</code></p></body></html>"
            )
        print("   ⚠ No articles found in database — wrote error page")
        return

    now_str = datetime.now().strftime("%d %B %Y, %H:%M")
    now_short = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Log available LLM providers (C5)
    providers = [name for name, key in [
        ("Gemini", GEMINI_API_KEY), ("Groq", GROQ_API_KEY),
        ("Mistral", MISTRAL_API_KEY), ("Cerebras", CEREBRAS_API_KEY),
    ] if key]
    print(f"   LLM providers: {', '.join(providers) if providers else 'NONE — briefs will fail'}")

    clusters = cluster_stories(df)
    n_multi = len([c for c in clusters if c["multi"] and not c["noise"]])

    # ── Assign each cluster to a country, 'regional', or 'internacional' ──
    # Only process multi-source, non-noise clusters (entity detection is expensive)
    display_clusters = [cl for cl in clusters if cl["multi"] and not cl["noise"]]
    country_clusters = {c: [] for c in COUNTRIES}
    regional_clusters = []
    internacional_clusters = []
    for cl in display_clusters:
        target = _cluster_country_assignment(cl)
        cl["country"] = target
        if target == "regional":
            regional_clusters.append(cl)
        elif target == "internacional":
            internacional_clusters.append(cl)
        elif target in country_clusters:
            country_clusters[target].append(cl)

    # ── Backfill country tabs with top single-source clusters when needed ──
    # When a country has fewer than PER_TAB_DISPLAY_CAP multi-source stories,
    # promote the best single-source clusters so the tab isn't near-empty.
    single_source_clusters = sorted(
        [cl for cl in clusters if not cl["multi"] and not cl["noise"]],
        key=lambda c: len(c["articles"]),
        reverse=True,
    )
    for country in COUNTRIES:
        deficit = PER_TAB_DISPLAY_CAP - len(country_clusters[country])
        if deficit <= 0:
            continue
        # Find single-source clusters whose articles are from this country
        # and are about domestic topics (not international news from local outlets)
        candidates = []
        for cl in single_source_clusters:
            if cl.get("country"):
                continue  # already assigned
            country_arts = [a for a in cl["articles"] if a.country == country]
            if len(country_arts) < len(cl["articles"]) * 0.8:
                continue
            # Skip international topics
            if _domestic_ratio(cl) < 0.5:
                continue
            # For single-article clusters, check title directly for foreign refs
            any_foreign = any(
                _FOREIGN_COUNTRY_PATTERNS.search(a.title) for a in cl["articles"]
            )
            any_local = any(
                _CONO_SUR_PATTERNS.search(a.title) for a in cl["articles"]
            )
            if any_foreign and not any_local:
                continue
            candidates.append(cl)
        # Sort by number of articles (bigger cluster = more coverage)
        candidates.sort(key=lambda c: len(c["articles"]), reverse=True)
        for cl in candidates[:deficit]:
            cl["country"] = country
            cl["multi"] = False  # keep flag so template can show "(single source)"
            country_clusters[country].append(cl)

    # Cap each tab's display list — top N by rank score (already sorted)
    for country in COUNTRIES:
        country_clusters[country] = country_clusters[country][:PER_TAB_DISPLAY_CAP]
    regional_clusters = regional_clusters[:PER_TAB_DISPLAY_CAP]
    internacional_clusters = internacional_clusters[:PER_TAB_DISPLAY_CAP]

    # ── Generate Summary+Context briefs for top N of each tab ──
    brief_targets = []
    for country in COUNTRIES:
        brief_targets.extend(country_clusters[country][:PER_TAB_BRIEF_CAP])
    brief_targets.extend(regional_clusters[:PER_TAB_BRIEF_CAP])
    brief_targets.extend(internacional_clusters[:PER_TAB_BRIEF_CAP])
    failed_briefs = []
    for cl in brief_targets:
        if cl.get("summary") is not None:
            continue  # already done (shouldn't happen, but defensive)
        summary, context = _generate_story_brief(cl["articles"])
        cl["summary"] = summary
        cl["context"] = context
        # Re-derive the card title from the summary so the headline above
        # the card can never contradict the summary body (e.g. a generic
        # "Milei y todas sus medidas" over a specific LIBRA-case summary).
        if summary:
            refined = _llm_synthesize_title_from_summary(summary)
            if refined:
                # Reject LLM title if it introduces foreign countries not in originals
                orig_foreign = set()
                for a in cl["articles"]:
                    m = _FOREIGN_COUNTRY_PATTERNS.findall(a.title)
                    orig_foreign.update(w.lower() for w in m)
                new_foreign = set(
                    w.lower() for w in _FOREIGN_COUNTRY_PATTERNS.findall(refined)
                )
                if not (new_foreign - orig_foreign):
                    cl["title"] = refined
        else:
            failed_briefs.append(cl)
        time.sleep(LLM_CALL_DELAY)

    # Retry failed briefs after a cooldown (rate limits may have cleared)
    if failed_briefs:
        time.sleep(LLM_RETRY_COOLDOWN)
        for cl in failed_briefs:
            summary, context = _generate_story_brief(cl["articles"])
            cl["summary"] = summary
            cl["context"] = context
            if summary:
                refined = _llm_synthesize_title_from_summary(summary)
                if refined:
                    orig_foreign = set()
                    for a in cl["articles"]:
                        m = _FOREIGN_COUNTRY_PATTERNS.findall(a.title)
                        orig_foreign.update(w.lower() for w in m)
                    new_foreign = set(
                        w.lower() for w in _FOREIGN_COUNTRY_PATTERNS.findall(refined)
                    )
                    if not (new_foreign - orig_foreign):
                        cl["title"] = refined
            else:
                # Fallback: synthesize a summary from headlines (B4)
                titles = list(dict.fromkeys(a.title for a in cl["articles"]))
                cl["summary"] = "Coverage: " + "; ".join(titles[:3])
            time.sleep(LLM_RETRY_DELAY)

    # ── Build country tabs ──
    used_top_titles = set()
    country_tabs_html = {}
    for country in COUNTRIES:
        top_title, _top_cluster = _pick_country_top_story(clusters, country, df, used_top_titles)
        flag = COUNTRY_FLAGS[country]
        name = COUNTRY_NAMES[country]

        stories_html = build_stories_html(
            country_clusters[country],
            empty_msg=f"No multi-source stories from {name} in this cycle.",
        )
        also_html = build_also_reported(df, clusters, country)
        n_country_stories = len(country_clusters[country])

        country_tabs_html[country] = f"""
        <div class="top-story-line">
            <span class="ts-label">Today's top story · {flag} {name}</span>
            <div class="ts-title">{_esc(top_title)}</div>
        </div>

        <div class="sec-head">Key Stories &mdash; {name} ({n_country_stories})</div>
        <p class="sec-desc">Multi-source stories where {name}'s press is the dominant voice. Each card shows how different newsrooms framed the same event.</p>
        <div class="stories-grid">{stories_html}</div>

        <div class="sec-head">Also Reported &mdash; {name}</div>
        <p class="sec-desc">Other noteworthy headlines from individual {name} outlets that didn't appear in multiple sources.</p>
        <div class="also-grid-single">{also_html}</div>
        """

    # ── Glossary: extract prominent people from headlines ──
    glossary_titles = []
    seen_titles = set()
    for cl in clusters:
        if cl["noise"]:
            continue
        for art in cl["articles"]:
            t = art.title.strip()
            if t and t not in seen_titles:
                seen_titles.add(t)
                glossary_titles.append(t)
        if len(glossary_titles) >= 120:  # cap prompt size
            break
    people_entries = _llm_extract_glossary(glossary_titles[:120])
    org_entries = _scan_titles_for_known_organizations(glossary_titles[:120])
    glossary_html = build_glossary_html(people_entries, org_entries)
    n_glossary = len(people_entries) + len(org_entries)

    # ── Build regional tab ──
    regional_stories_html = build_stories_html(
        regional_clusters,
        group_by_country=True,
        empty_msg="No cross-border stories in this cycle.",
    )
    n_regional = len(regional_clusters)
    flags_all = (
        COUNTRY_FLAGS["argentina"] + " "
        + COUNTRY_FLAGS["uruguay"] + " "
        + COUNTRY_FLAGS["paraguay"]
    )
    regional_tab_html = f"""
        <div class="top-story-line">
            <span class="ts-label">Cross-border coverage · {flags_all}</span>
            <div class="ts-title">Stories picked up by the press of two or more countries.</div>
        </div>

        <div class="sec-head">Regional Stories ({n_regional})</div>
        <p class="sec-desc">Events covered by newsrooms from multiple Cono Sur countries. Source lines are grouped by country so you can compare how each national press framed the same event.</p>
        <div class="stories-grid">{regional_stories_html}</div>
    """

    # ── Build internacional tab ──
    intl_stories_html = build_stories_html(
        internacional_clusters,
        group_by_country=True,
        empty_msg="No international stories in this cycle.",
    )
    n_internacional = len(internacional_clusters)
    internacional_tab_html = f"""
        <div class="top-story-line">
            <span class="ts-label">International coverage · \U0001F30D</span>
            <div class="ts-title">Global events covered by Cono Sur media but not primarily about the region.</div>
        </div>

        <div class="sec-head">International Stories ({n_internacional})</div>
        <p class="sec-desc">Events from outside the Cono Sur that received multi-source coverage in the region's press.</p>
        <div class="stories-grid">{intl_stories_html}</div>
    """

    total = len(df)
    sources = df["source"].nunique()

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Media Monitor &mdash; Argentina, Paraguay &amp; Uruguay &mdash; {now_short}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: #FDF0D5;
    color: #003049;
    line-height: 1.55;
    font-size: 13px;
    max-width: 100%;
    margin: 0;
    padding: 0 32px;
}}

/* ── Letterhead ── */
.letterhead {{
    background: #FDF0D5;
    padding: 16px 0 12px;
    color: #003049;
    border-bottom: 1px solid rgba(0,48,73,0.12);
}}
.lh-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.lh-title {{
    font-size: 15px;
    font-weight: 700;
    color: #003049;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}}
.lh-sub {{
    font-size: 10px;
    color: rgba(0,48,73,0.60);
    font-weight: 600;
    letter-spacing: 1px;
    margin-top: 2px;
}}
.lh-date {{
    font-size: 10px;
    color: rgba(0,48,73,0.65);
    text-align: right;
    font-family: ui-monospace, 'Cascadia Code', 'Consolas', monospace;
    font-weight: 700;
}}

/* ── Content ── */
.content {{
    padding: 10px 0 12px;
    background: #FDF0D5;
}}
.stories-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
}}
.also-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}}

/* ── Section headers ── */
.sec-head {{
    font-size: 15px;
    font-weight: 700;
    color: #003049;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 14px 0 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid rgba(0,48,73,0.12);
}}
.sec-head:first-child {{ margin-top: 4px; }}
.sec-desc {{
    font-size: 12px;
    color: #003049;
    margin: 2px 0 8px;
    padding: 4px 8px;
    background: rgba(0,48,73,0.04);
    border: 1px solid rgba(0,48,73,0.08);
    border-radius: 4px;
    line-height: 1.4;
}}

/* ── Tabs ── */
.tab-nav {{
    display: flex;
    gap: 4px;
    margin: 12px 0 0;
    border-bottom: 2px solid rgba(0,48,73,0.18);
}}
.tab-btn {{
    appearance: none;
    background: rgba(0,48,73,0.04);
    border: 1px solid rgba(0,48,73,0.10);
    border-bottom: none;
    color: rgba(0,48,73,0.65);
    font-family: inherit;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    padding: 7px 14px;
    cursor: pointer;
    border-radius: 4px 4px 0 0;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-bottom: -2px;
    transition: background 0.15s, color 0.15s;
}}
.tab-btn:hover {{
    background: rgba(0,48,73,0.10);
    color: #003049;
}}
.tab-btn.active {{
    background: #003049;
    color: #FDF0D5;
    border-color: #003049;
}}
.tab-btn.active .flag {{
    filter: brightness(1.05);
}}
.tab-btn .tab-count {{
    font-size: 9px;
    font-weight: 600;
    background: rgba(255,255,255,0.20);
    padding: 1px 5px;
    border-radius: 9999px;
    font-family: ui-monospace, 'Cascadia Code', 'Consolas', monospace;
}}
.tab-btn:not(.active) .tab-count {{
    background: rgba(0,48,73,0.10);
}}
.tab-panel {{
    display: none;
    padding-top: 6px;
}}
.tab-panel.active {{
    display: block;
}}

/* ── Top story line (per tab) ── */
.top-story-line {{
    margin: 8px 0 4px;
    padding: 8px 12px;
    background: rgba(0,48,73,0.06);
    border: 1px solid rgba(0,48,73,0.10);
    border-left: 3px solid #1a6fa3;
    border-radius: 4px;
}}
.ts-label {{
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: rgba(0,48,73,0.60);
}}
.ts-title {{
    font-size: 13px;
    font-weight: 600;
    color: #003049;
    margin-top: 2px;
    line-height: 1.4;
}}

/* ── Country group (Regional tab) ── */
.country-group {{
    margin: 4px 0 2px;
    padding: 3px 0 3px 6px;
    border-left: 2px solid rgba(0,48,73,0.18);
}}
.cg-header {{
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: rgba(0,48,73,0.60);
    margin-bottom: 2px;
    display: flex;
    align-items: center;
    gap: 4px;
}}

.also-grid-single {{
    display: block;
}}

/* ── Glossary (Quién es quién) ── */
.gloss-section {{
    margin: 10px 0 14px;
}}
.gloss-section-header {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: rgba(0,48,73,0.65);
    padding-bottom: 4px;
    margin-bottom: 6px;
    border-bottom: 1px solid rgba(0,48,73,0.12);
    display: flex;
    align-items: center;
    gap: 6px;
}}
.gloss-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
}}
.gloss-card {{
    padding: 8px 10px;
    background: rgba(0,48,73,0.04);
    border: 1px solid rgba(0,48,73,0.10);
    border-left: 3px solid #669BBC;
    border-radius: 4px;
}}
.gloss-name {{
    font-size: 12px;
    font-weight: 700;
    color: #003049;
    line-height: 1.3;
}}
.gloss-role {{
    font-size: 10px;
    font-weight: 600;
    color: rgba(0,48,73,0.70);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 1px;
}}
.gloss-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 5px;
}}
.meta-pill {{
    display: inline-block;
    padding: 1px 6px;
    border-radius: 9999px;
    background: rgba(0,48,73,0.06);
    color: rgba(0,48,73,0.72);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}}
.gloss-bio {{
    font-size: 11px;
    color: rgba(0,48,73,0.78);
    line-height: 1.4;
    margin-top: 4px;
    padding-top: 4px;
    border-top: 1px dashed rgba(0,48,73,0.15);
    font-style: italic;
}}

/* ── Stories ── */
.story {{
    margin-bottom: 10px;
    padding: 8px 12px;
    background: rgba(0,48,73,0.04);
    border: 1px solid rgba(0,48,73,0.08);
    border-radius: 4px;
}}
.story-header {{
    display: flex;
    align-items: center;
    gap: 5px;
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
    font-family: ui-monospace, 'Cascadia Code', 'Consolas', monospace;
}}
.story-sources {{
    font-size: 9px;
    color: rgba(0,48,73,0.65);
    margin-left: auto;
    font-family: ui-monospace, 'Cascadia Code', 'Consolas', monospace;
    font-weight: 700;
}}
.story-title {{
    font-size: 13px;
    font-weight: 700;
    color: #003049;
    line-height: 1.35;
    margin-bottom: 3px;
    padding-bottom: 3px;
    border-bottom: 1px solid rgba(0,48,73,0.12);
}}
.story-summary {{
    font-size: 12px;
    color: #003049;
    line-height: 1.45;
    margin: 3px 0 4px 0;
    padding: 5px 8px;
    background: rgba(0,48,73,0.08);
    border-left: 3px solid #1a6fa3;
    border-radius: 2px;
}}
.story-context {{
    font-size: 11px;
    color: rgba(0,48,73,0.78);
    line-height: 1.4;
    margin: 0 0 5px 0;
    padding: 4px 8px;
    border-left: 2px solid rgba(0,48,73,0.30);
    background: rgba(0,48,73,0.03);
    border-radius: 2px;
    font-style: italic;
}}
.ctx-label {{
    display: inline-block;
    font-style: normal;
    font-weight: 700;
    font-size: 8px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: rgba(0,48,73,0.55);
    margin-right: 6px;
    padding: 1px 5px;
    border: 1px solid rgba(0,48,73,0.20);
    border-radius: 3px;
    vertical-align: 1px;
}}
.story-body {{
    padding: 1px 0;
}}
.story-variant {{
    display: flex;
    gap: 6px;
    padding: 1px 0;
    font-size: 11px;
    line-height: 1.35;
    align-items: baseline;
}}
.sv-source {{
    font-weight: 700;
    color: rgba(0,48,73,0.65);
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    min-width: 90px;
    flex-shrink: 0;
}}
.sv-title {{
    flex: 1;
    color: #003049;
    text-decoration: none;
}}
.sv-title:hover {{
    text-decoration: underline;
}}
.story-variant:hover {{
    background: rgba(0,48,73,0.06);
    border-radius: 2px;
}}

/* ── Badge ── */
.badge {{
    display: inline-block;
    box-sizing: border-box;
    padding: 2px 8px;
    min-width: 46px;
    border-radius: 9999px;
    color: white;
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    text-align: center;
    white-space: nowrap;
    vertical-align: middle;
    border: 1px solid rgba(0,48,73,0.15);
}}
.sub-badge {{
    display: inline-block;
    box-sizing: border-box;
    padding: 2px 8px;
    border-radius: 9999px;
    background: rgba(0,48,73,0.08);
    color: rgba(0,48,73,0.78);
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    white-space: nowrap;
    vertical-align: middle;
    border: 1px solid rgba(0,48,73,0.12);
}}

/* ── Coverage Dots ── */
.coverage-dots {{
    display: inline-flex;
    gap: 2px;
    vertical-align: middle;
    margin-left: 3px;
}}
.cov-dot {{
    width: 5px;
    height: 5px;
    border-radius: 50%;
    display: inline-block;
}}

/* ── Stats Bar ── */
.stats-row {{
    display: flex;
    gap: 16px;
    margin: 8px 0 4px;
    padding: 8px 12px;
    background: rgba(0,48,73,0.04);
    border-radius: 4px;
    border: 1px solid rgba(0,48,73,0.08);
}}
.stats-group {{
    flex: 1;
}}
.stats-label {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: rgba(0,48,73,0.65);
    margin-bottom: 3px;
}}
.stat-item {{
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 1px 0;
}}
.stat-bar {{
    flex: 1;
    height: 5px;
    background: rgba(0,48,73,0.10);
    border-radius: 9999px;
    overflow: hidden;
}}
.stat-fill {{
    height: 100%;
    border-radius: 9999px;
}}
.stat-num {{
    font-size: 13px;
    font-weight: 700;
    color: #003049;
    font-family: ui-monospace, 'Cascadia Code', 'Consolas', monospace;
    min-width: 24px;
    text-align: right;
}}

/* ── Also Reported ── */
.also-section {{
    margin-bottom: 8px;
}}
.also-header {{
    font-size: 11px;
    font-weight: 600;
    color: rgba(0,48,73,0.65);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 3px 0;
    border-bottom: 1px solid rgba(0,48,73,0.12);
    margin-bottom: 2px;
}}
.also-item {{
    display: flex;
    gap: 5px;
    padding: 2px 0;
    font-size: 11px;
    line-height: 1.35;
    align-items: baseline;
    border-bottom: 1px solid rgba(0,48,73,0.06);
}}
.also-item:last-child {{ border-bottom: none; }}
.also-date {{ color: rgba(0,48,73,0.60); font-size: 10px; min-width: 32px; font-family: ui-monospace, 'Cascadia Code', 'Consolas', monospace; }}
.also-source {{ font-weight: 600; color: rgba(0,48,73,0.65); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; min-width: 85px; flex-shrink: 0; }}
.also-title {{ flex: 1; color: #003049; text-decoration: none; }}
.also-title:hover {{ text-decoration: underline; }}

/* ── Footer ── */
.doc-footer {{
    background: #FDF0D5;
    padding: 8px 0;
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    color: rgba(0,48,73,0.60);
    font-family: ui-monospace, 'Cascadia Code', 'Consolas', monospace;
    border-top: 1px solid rgba(0,48,73,0.12);
}}

.muted {{ color: rgba(0,48,73,0.60); font-style: italic; }}
a {{ color: inherit; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.flag {{ width: 14px; height: 14px; vertical-align: middle; display: inline-block; }}

/* ── Mobile ── */
@media (max-width: 768px) {{
    body {{ padding: 0 12px; }}
    .lh-top {{ flex-direction: column; align-items: flex-start; gap: 4px; }}
    .lh-date {{ text-align: left; }}
    .stories-grid {{ grid-template-columns: 1fr; }}
    .gloss-grid {{ grid-template-columns: 1fr; }}
    .tab-nav {{ flex-wrap: wrap; }}
    .tab-btn {{ flex: 1 1 auto; padding: 6px 8px; font-size: 10px; letter-spacing: 0.8px; }}
    .also-grid {{ grid-template-columns: 1fr; }}
    .story-header {{ gap: 4px; }}
    .story-variant {{ flex-wrap: wrap; }}
    .sv-source {{ min-width: 70px; }}
    .also-item {{ flex-wrap: wrap; }}
    .stats-row {{ flex-direction: column; gap: 8px; }}
}}

/* ── Print ── */
@media print {{
    body {{ font-size: 10.5px; max-width: none; background: white; }}
    .letterhead {{ padding: 12px 20px 8px; }}
    .content {{ padding: 6px 20px 8px; }}
    .story {{ break-inside: avoid; }}
    .also-section {{ break-inside: avoid; }}
    .letterhead, .badge, .story-num, .doc-footer {{
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }}
    /* Show all tabs when printing */
    .tab-nav {{ display: none; }}
    .tab-panel {{ display: block !important; page-break-before: auto; }}
    @page {{ margin: 1cm; size: A4 portrait; }}
}}
</style>
</head>
<body>

<!-- Letterhead -->
<div class="letterhead">
    <div class="lh-top">
        <div>
            <div class="lh-title">MEDIA MONITOR — Argentina, Paraguay &amp; Uruguay</div>
            <div class="lh-sub">{COUNTRY_FLAGS['argentina']} Argentina &middot; {COUNTRY_FLAGS['uruguay']} Uruguay &middot; {COUNTRY_FLAGS['paraguay']} Paraguay</div>
        </div>
        <div class="lh-date">{now_str}</div>
    </div>
</div>

<div class="content">

    <nav class="tab-nav" role="tablist">
        <button class="tab-btn active" data-tab="tab-ar" role="tab">{COUNTRY_FLAGS['argentina']} Argentina <span class="tab-count">{len(country_clusters['argentina'])}</span></button>
        <button class="tab-btn" data-tab="tab-uy" role="tab">{COUNTRY_FLAGS['uruguay']} Uruguay <span class="tab-count">{len(country_clusters['uruguay'])}</span></button>
        <button class="tab-btn" data-tab="tab-py" role="tab">{COUNTRY_FLAGS['paraguay']} Paraguay <span class="tab-count">{len(country_clusters['paraguay'])}</span></button>
        <button class="tab-btn" data-tab="tab-regional" role="tab">Regional <span class="tab-count">{n_regional}</span></button>
        <button class="tab-btn" data-tab="tab-intl" role="tab">\U0001F30D Internacional <span class="tab-count">{n_internacional}</span></button>
        <button class="tab-btn" data-tab="tab-glossary" role="tab">Actores Clave <span class="tab-count">{n_glossary}</span></button>
    </nav>

    <section id="tab-ar" class="tab-panel active" role="tabpanel">
        {country_tabs_html['argentina']}
    </section>

    <section id="tab-uy" class="tab-panel" role="tabpanel">
        {country_tabs_html['uruguay']}
    </section>

    <section id="tab-py" class="tab-panel" role="tabpanel">
        {country_tabs_html['paraguay']}
    </section>

    <section id="tab-regional" class="tab-panel" role="tabpanel">
        {regional_tab_html}
    </section>

    <section id="tab-intl" class="tab-panel" role="tabpanel">
        {internacional_tab_html}
    </section>

    <section id="tab-glossary" class="tab-panel" role="tabpanel">
        <div class="top-story-line">
            <span class="ts-label">Actores Clave · Cono Sur</span>
            <div class="ts-title">People and institutions most repeatedly mentioned across the region's press in this cycle.</div>
        </div>
        <p class="sec-desc">Entities are extracted automatically from today's headlines and ranked with mention count plus confidence signals. Bios are factual context, not editorial commentary.</p>
        {glossary_html}
    </section>

</div>

<script>
(function() {{
    var buttons = document.querySelectorAll('.tab-btn');
    var panels = document.querySelectorAll('.tab-panel');
    buttons.forEach(function(btn) {{
        btn.addEventListener('click', function() {{
            var target = btn.getAttribute('data-tab');
            buttons.forEach(function(b) {{ b.classList.remove('active'); }});
            panels.forEach(function(p) {{ p.classList.remove('active'); }});
            btn.classList.add('active');
            var panel = document.getElementById(target);
            if (panel) panel.classList.add('active');
        }});
    }});
}})();
</script>

<div class="doc-footer">
    <span>MEDIA MONITOR — Argentina, Paraguay &amp; Uruguay</span>
    <span>{now_short} | {total} articles, {sources} sources</span>
</div>

</body>
</html>"""

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    # Close Playwright browser if it was used for body extraction
    from scraper import _close_playwright
    _close_playwright()

    print(f"   Dashboard generado: {OUTPUT_PATH}")
    print(f"      {total} articulos, {sources} fuentes, {n_multi} multi-source stories")


if __name__ == "__main__":
    generate_dashboard()

