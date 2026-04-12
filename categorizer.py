"""Article filtering and classification helpers for the media monitor."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

SOURCE_RULES = {
    "Cronica PY": {
        "deny_title_patterns": [
            r"\bvip\b", r"\bvideo\b", r"\bfarandula\b", r"\bshow\b",
            r"\balbirrojo\b", r"\bportero\b", r"\btatuaje\b",
        ],
        "deny_url_terms": ("deportes", "fama", "show", "mbeyu", "viral"),
        "strict_min_tokens": 5,
    },
    "5Dias": {
        "deny_title_patterns": [r"\blifestyle\b", r"\bcolumnas\b", r"\bopinion\b"],
        "deny_url_terms": ("lifestyle", "columnas", "opinion"),
    },
    "Hoy": {
        "deny_title_patterns": [r"\bvideo\b", r"\bshow\b", r"\bviral\b"],
        "deny_url_terms": ("espectaculos", "show", "viral"),
    },
    "Ultima Hora": {
        "deny_url_terms": ("espectaculos", "deportes", "tv"),
    },
}

_IRRELEVANT_PATTERNS = [
    r"\bhoroscopo\b",
    r"\bquiniela\b",
    r"\bquini ?6\b",
    r"\bloteria\b",
    r"\btelekino\b",
    r"\bbrinco\b",
    r"\btombola\b",
    r"\breceta\b",
    r"\bshow\b",
    r"\bespectaculos\b",
    r"\bfarandula\b",
    r"\bdeportes?\b",
    r"\bfutbol\b",
    r"\btenis\b",
    r"\bbasquet\b",
    r"\bchampions\b",
    r"\bgaleria\b",
    r"\bsuscrib",
    r"\bnewsletter\b",
    r"\bpodcast\b",
    r"\blifestyle\b",
    r"\bviral\b",
    r"\bmbeju\b",
    r"\bmbeyu\b",
    # Sports match results that bypass the "deportes" keyword
    r"\b\d+\s*-\s*\d+\b.*\b(gol|penales?|apertura|clausura|fecha\s+\d)\b",
    r"\bcampeon\b",
    r"\bse juega\b.*\b(primer|segundo)\b",
    r"\bdebut[oó]?\b.*\bprimera\b",
    r"\bgol de\b",
    r"\bcopa libertadores\b",
    r"\bcopa america\b",
    r"\bcopa sudamericana\b",
    r"\bsupercopa\b",
    r"\b(penarol|nacional|cerro porteno|olimpia)\b.*(gano|perdio|empato|vencio)",
    r"\bdeportivo maldonado\b",
    r"\bciclismo\b",
    r"\b\d+\s*fotos?\b",
]

_CATEGORY_KEYWORDS = {
    "politica": [
        ("presidente", 3.0), ("vicepresidente", 3.0), ("gobierno", 2.6),
        ("ministro", 2.5), ("senado", 2.4), ("senador", 2.2),
        ("diputado", 2.1), ("congreso", 2.5), ("parlamento", 2.5),
        ("gabinete", 2.2), ("decreto", 2.1), ("ley", 1.8),
        ("eleccion", 3.0), ("electoral", 2.8), ("oposicion", 1.8),
        ("oficialismo", 1.8), ("coalicion", 1.7), ("partido", 1.6),
        ("intendente", 2.2), ("gobernador", 2.2), ("legislador", 1.8),
        ("corte suprema", 1.8), ("fiscal", 1.6), ("juez", 1.6),
        ("milei", 2.8), ("orsi", 2.8), ("pena", 2.5), ("cartes", 2.1),
        ("lacalle", 2.2), ("frente amplio", 2.0), ("honor colorado", 1.8),
    ],
    "economia": [
        ("economia", 3.0), ("inflacion", 3.0), ("dolar", 3.0),
        ("banco central", 2.8), ("bcra", 2.8), ("fmi", 2.7),
        ("deuda", 2.5), ("reservas", 2.3), ("salario", 2.0),
        ("empleo", 2.0), ("mercado", 1.8), ("export", 2.2),
        ("import", 2.2), ("inversion", 2.1), ("impuesto", 2.1),
        ("presupuesto", 2.0), ("tarifa", 2.0), ("combustible", 2.0),
        ("petroleo", 1.8), ("gas", 1.5), ("energia", 1.7),
        ("empresa", 1.4), ("industria", 1.6), ("proveedores", 1.6),
        ("licitacion", 1.6), ("constructoras", 1.8), ("obras", 1.5),
        ("yerbatero", 2.4), ("sector yerbatero", 2.6), ("ganadera", 1.7),
        ("agronegocios", 1.8), ("mef", 1.8), ("petropar", 2.0),
        ("mercosur", 1.8), ("mercados", 1.8), ("produccion", 1.8),
        ("productores", 1.8), ("innovacion", 1.5), ("exoneracion", 1.8),
        ("multas", 1.4), ("laborales", 1.7), ("obligaciones laborales", 2.4),
    ],
    "internacional": [
        ("estados unidos", 3.0), ("eeuu", 2.8), ("china", 2.7),
        ("rusia", 2.7), ("ucrania", 2.7), ("israel", 2.7),
        ("gaza", 2.7), ("onu", 2.3), ("oea", 2.1), ("union europea", 2.6),
        ("ue", 2.1), ("mercosur ue", 2.6), ("mercosur-ue", 2.6),
        ("cumbre", 2.0), ("bilateral", 2.2), ("embajada", 2.0),
        ("embajador", 2.0), ("canciller", 2.0), ("diplomacia", 2.1),
        ("cooperacion", 1.7), ("conflicto", 1.6), ("guerra", 2.2),
        ("oriente medio", 3.0), ("visita oficial", 2.1), ("regional", 1.4),
        ("panama", 1.1), ("japon", 1.1), ("taiwan", 1.1), ("argelia", 1.1),
    ],
}

_URL_HINTS = {
    "politica": ("politica", "nacional", "gobierno", "congreso", "presidencia"),
    "economia": ("economia", "negocios", "dinero", "finanzas", "empresas"),
    "internacional": ("mundo", "internacional", "exterior", "world"),
}

_SUBTOPIC_RULES = {
    "economia": {
        "macroeconomia": [
            "inflacion", "dolar", "reservas", "banco central", "fmi",
            "deuda", "tasas", "presupuesto", "deficit",
        ],
        "energia": ["energia", "gas", "petroleo", "combustible", "ypf", "petropar"],
        "agro": ["yerbatero", "ganadera", "agronegocios", "soja", "trigo", "maiz"],
        "comercio": ["export", "import", "mercosur", "mercados", "inversion", "arancel"],
        "obra_publica": ["constructoras", "licitacion", "obras", "mopc", "infraestructura"],
        "empleo": ["empleo", "laborales", "salario", "trabajo", "proveedores"],
    },
    "politica": {
        "ejecutivo": ["presidente", "vicepresidente", "gabinete", "gobierno", "ministro"],
        "legislativo": ["senado", "senador", "diputado", "congreso", "parlamento", "ley"],
        "electoral": ["eleccion", "electoral", "campana", "partido", "coalicion"],
        "justicia": ["corte suprema", "fiscal", "juez", "tribunal", "causa"],
        "seguridad": ["interior", "seguridad", "policia", "narco", "crimen"],
        "social": ["salud", "educacion", "ips", "programas sociales", "asegurados"],
    },
    "internacional": {
        "diplomacia": ["bilateral", "cumbre", "embajada", "embajador", "canciller", "cooperacion"],
        "comercio_exterior": ["mercosur", "inversion", "mercados", "comercio", "export", "import"],
        "conflictos": ["guerra", "conflicto", "gaza", "ucrania", "israel", "rusia"],
        "multilateral": ["onu", "oea", "g20", "g7", "union europea"],
    },
}


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def _normalize(text: str) -> str:
    text = _strip_accents((text or "").lower())
    text = re.sub(r"[^a-z0-9/\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalized_url_path(url: str) -> str:
    if not url:
        return ""
    path = urlparse(url).path.replace("-", " ").replace("/", " ")
    return _normalize(path)


def _matches_keyword(text: str, keyword: str) -> bool:
    keyword = _normalize(keyword)
    if not keyword:
        return False
    if " " in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\w*", text) is not None


def _score_keywords(text: str, keywords: list[tuple[str, float]]) -> float:
    return sum(weight for keyword, weight in keywords if _matches_keyword(text, keyword))


def _source_rule_value(source: str, key: str, default):
    rule = SOURCE_RULES.get(source or "", {})
    return rule.get(key, default)


def is_relevant_headline(title: str, url: str = "", source: str = "") -> bool:
    """Return False for obvious junk, promo, and off-topic headlines."""
    norm_title = _normalize(title)
    norm_url = _normalized_url_path(url)
    if len(norm_title) < 24:
        return False
    if any(re.search(pattern, norm_title) for pattern in _IRRELEVANT_PATTERNS):
        return False
    if any(re.search(pattern, norm_title) for pattern in _source_rule_value(source, "deny_title_patterns", [])):
        return False
    if any(term in norm_url for term in _source_rule_value(source, "deny_url_terms", ())):
        return False
    alpha_tokens = [tok for tok in norm_title.split() if tok.isalpha() and len(tok) >= 3]
    min_tokens = _source_rule_value(source, "strict_min_tokens", 4)
    if len(alpha_tokens) < min_tokens:
        return False
    if "/video" in url.lower() or "autor" in norm_url or "newsletter" in norm_url:
        return False
    return True


def infer_category(title: str, url: str = "", fallback: str | None = None) -> tuple[str | None, float]:
    """Infer a broad category from title and URL."""
    norm_title = _normalize(title)
    norm_url = _normalized_url_path(url)
    haystack = f"{norm_title} {norm_url}".strip()
    scores = {category: 0.0 for category in _CATEGORY_KEYWORDS}

    for category, keywords in _CATEGORY_KEYWORDS.items():
        scores[category] += _score_keywords(haystack, keywords)
        for hint in _URL_HINTS[category]:
            if hint in norm_url:
                scores[category] += 1.5

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_category, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = round(best_score - second_score, 2)

    if best_score < 1.5:
        return fallback, 0.0
    if confidence < 0.6 and fallback:
        return fallback, confidence
    return best_category, confidence


def infer_subcategory(title: str, url: str = "", category: str | None = None) -> str | None:
    """Infer a narrower subtopic inside the chosen broad category."""
    if not category or category not in _SUBTOPIC_RULES:
        return None
    norm_title = _normalize(title)
    norm_url = _normalized_url_path(url)
    haystack = f"{norm_title} {norm_url}".strip()
    best_name = None
    best_score = 0
    for subtopic, keywords in _SUBTOPIC_RULES[category].items():
        score = sum(1 for keyword in keywords if _matches_keyword(haystack, keyword))
        if score > best_score:
            best_name = subtopic
            best_score = score
    return best_name if best_score > 0 else None


def analyze_article(
    title: str,
    url: str = "",
    source: str = "",
    fallback: str | None = None,
    strict: bool = False,
) -> dict:
    """Return relevance, broad category, subcategory, and confidence signals."""
    if not is_relevant_headline(title, url, source):
        return {
            "accepted": False,
            "category": None,
            "subcategory": None,
            "confidence": 0.0,
            "reason": "irrelevant",
        }

    category, confidence = infer_category(title, url, fallback=fallback)
    if not category:
        return {
            "accepted": False,
            "category": None,
            "subcategory": None,
            "confidence": confidence,
            "reason": "unclear-category",
        }

    if strict and confidence < 0.75 and category == fallback:
        return {
            "accepted": False,
            "category": category,
            "subcategory": infer_subcategory(title, url, category),
            "confidence": confidence,
            "reason": "low-confidence-html",
        }

    subcategory = infer_subcategory(title, url, category)
    return {
        "accepted": True,
        "category": category,
        "subcategory": subcategory,
        "confidence": float(confidence),
        "reason": "ok",
    }


def classify_article(
    title: str,
    url: str = "",
    fallback: str | None = None,
    strict: bool = False,
    source: str = "",
) -> str | None:
    """Return the accepted category for an article, or None if it looks noisy."""
    result = analyze_article(title, url, source=source, fallback=fallback, strict=strict)
    return result["category"] if result["accepted"] else None


def categorize(title: str) -> str:
    """Backward-compatible wrapper used by older callers."""
    category, _confidence = infer_category(title)
    return category or "otros"
