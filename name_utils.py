"""Shared name-validation utilities for the media monitor pipeline.

Used by both ``qeq_loader.py`` (parsing Excel roster) and ``dashboard.py``
(entity detection at runtime).  Keeping a single definition prevents the
two copies from diverging.
"""

from __future__ import annotations

import re

# Connectors that appear between given and family names — not meaningful
# words for deciding whether a string is a person name.
NAME_CONNECTORS = {
    "da", "de", "del", "de la", "de las", "de los", "do", "dos", "das",
    "y", "e", "van", "von",
}

# Tokens that signal institutions, parties, brands, or concepts — never
# part of a real person's name in the QeQ roster or in headlines.
NON_PERSON_TOKENS = {
    "administracion", "agencia", "alianza", "ande", "antimafia", "armadas",
    "asociacion", "autoridad", "banco", "bloque", "camara", "capital",
    "cartel", "centro", "clan", "club", "coalicion", "comando", "comision",
    "comite", "comunidad", "confederacion", "congreso", "consejo",
    "coordinadora", "corte", "cruzada", "defensa", "diario", "direccion",
    "ejercito", "empresa", "estado", "familia", "familiares", "federacion",
    "fiscalia", "frente", "fundacion", "gobierno", "grupo", "hermanos",
    "hora", "hub", "abuelas", "hijos", "iglesia", "instituto", "junta",
    "las", "leonas", "leones", "lista", "los", "madre", "madres", "medio",
    "mercosur", "ministerio", "movimiento", "municipalidad", "nacion",
    "nomadas", "observatorio", "organismo", "organizacion", "panteras",
    "paraguay", "argentina", "uruguay", "partido", "periodico", "policia",
    "presidencia", "primer", "programa", "pumas", "republica", "secretaria",
    "seleccion", "senado", "servicio", "sindicato", "sistema", "sociedad",
    "suprema", "television", "teros", "tierra", "tribunal", "ultima",
    "unidad", "universidad",
}


def strip_accents(s: str) -> str:
    """Lowercase and replace common Spanish/Portuguese diacritics."""
    return (
        s.lower()
        .replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
        .replace("ü", "u")
    )


def normalize_person_name(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace for name matching."""
    if not name:
        return ""
    s = name.lower().strip()
    accent_map = str.maketrans("áéíóúñü", "aeiounu")
    s = s.translate(accent_map)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


def looks_like_person_name(name: str) -> bool:
    """Reject institutions, parties, media brands, and concept phrases.

    Accepts strings that look like a real person's full name (2-6 tokens,
    at least two capitalised substantial words, no institutional keywords).
    """
    if not name or re.search(r"\d", name) or "/" in str(name):
        return False
    cleaned = re.sub(r"[^\w\s'.-]", " ", str(name)).strip()
    tokens = [tok for tok in re.split(r"\s+", cleaned) if tok]
    if len(tokens) < 2 or len(tokens) > 6:
        return False

    substantial: list[str] = []
    uppercase_like = 0
    for token in tokens:
        norm = strip_accents(token)
        if norm in NAME_CONNECTORS:
            continue
        if not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", token):
            return False
        if norm in NON_PERSON_TOKENS:
            return False
        if len(token) > 1 and token.isupper():
            return False
        substantial.append(token)
        if token[0].isupper():
            uppercase_like += 1

    if len(substantial) < 2:
        return False
    if uppercase_like < 2:
        return False
    return True
