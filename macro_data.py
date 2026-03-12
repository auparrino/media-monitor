"""Macroeconomic indicators with API fetching and hardcoded fallback."""

import requests

FALLBACK = {
    "argentina": {
        "usd_oficial": 1065.0,
        "inflacion_mensual": 2.4,
        "inflacion_anual": 66.9,
        "reservas_bcra_bn": 28.5,
        "fecha": "marzo 2026",
    },
    "uruguay": {
        "usd": 41.8,
        "inflacion_anual": 5.1,
        "fecha": "marzo 2026",
    },
    "paraguay": {
        "usd": 7820.0,
        "inflacion_anual": 4.2,
        "fecha": "marzo 2026",
    },
}


def _try_bcra_usd():
    """Try to fetch USD oficial from BCRA API."""
    try:
        url = "https://api.bcra.gob.ar/estadisticas/v2.0/DatosVariable/1/2026-02-01/2026-03-07"
        resp = requests.get(url, timeout=10, verify=False)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                latest = results[-1]
                return float(latest.get("valor", 0))
    except Exception:
        pass
    return None


def get_macro_data():
    """Return macro data dict. Always returns data, never crashes."""
    data = {k: dict(v) for k, v in FALLBACK.items()}

    # Try live BCRA data for Argentina
    bcra_usd = _try_bcra_usd()
    if bcra_usd and bcra_usd > 0:
        data["argentina"]["usd_oficial"] = bcra_usd
        data["argentina"]["fecha"] = "live (BCRA)"

    return data


if __name__ == "__main__":
    import json
    print(json.dumps(get_macro_data(), indent=2, ensure_ascii=False))
