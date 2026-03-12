"""Categorize news article titles by keyword matching."""

import re

CATEGORIES = {
    "economia": [
        "inflaci", "tipo de cambio", "dólar", "dolar", "peso", "rigi",
        "inversión", "inversion", "fmi", "deuda", "pbi", "reservas", "bcra",
        "desempleo", "salario", "presupuesto", "déficit", "deficit",
        "superávit", "superavit", "balanza", "devaluaci", "ajuste",
        "tarifas", "energía", "energia", "combustible", "gas", "petróleo",
        "petroleo", "ypf", "vaca muerta", "soja", "bolsa", "bonos",
        "riesgo país", "riesgo pais", "privatizaci", "reforma tributaria",
        "recaudaci", "impuesto", "tasa de interés", "tasa de interes",
    ],
    "politica": [
        "elecci", "gobierno", "congreso", "parlamento", "partido",
        "presidente", "senado", "diputad", "oposición", "oposicion",
        "coalición", "coalicion", "gabinete", "ministro", "intendente",
        "gobernador", "milei", "orsi", "petta", "lacalle", "santiago peña",
        "santiago pena", "lla", "frente amplio", "pro ", "peronismo",
        "legislat", "decreto", "constituc",
    ],
    "relaciones_exteriores": [
        "cancillería", "cancilleria", "diplomacia", "embajada", "acuerdo bilateral",
        "tratado", "oea", " onu", "cumbre", "visita oficial", "canciller",
        "australia", "china", "eeuu", "estados unidos", "rusia", "europa",
        "mercosur-ue", "mercosur ue", "bilateral", "g20", "g7",
    ],
    "comercio": [
        "exportaci", "importaci", "mercosur", "arancel", "litio",
        "minería", "mineria", "cobre", "oro ", "agro", "cereal", "trigo",
        "maíz", "maiz", "carne", "inversión extranjera", "inversion extranjera",
        "libre comercio", "balanza comercial",
    ],
    "seguridad": [
        "narcotráfico", "narcotrafico", "crimen", "policía", "policia",
        "fuerzas armadas", "delito", "homicidio", "inseguridad", "narco",
        "cartel", "corrupción", "corrupcion",
    ],
}


def categorize(title: str) -> str:
    """Return the first matching category for a title, or 'otros'."""
    title_lower = title.lower()
    for category, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in title_lower:
                return category
    return "otros"
