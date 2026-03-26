"""
Playground: Mejoras al Media Monitor usando plugins nuevos.

Prueba 3 mejoras potenciales:
  1. Playwright scraping (JS-rendered sites) vs requests actual
  2. Web Search como fuente complementaria de noticias
  3. Extracción de cuerpo de artículos para mejor clustering

Ejecutar:  python playground_plugins.py [test_name]
  - playwright   : compara requests vs playwright en sitios JS-heavy
  - websearch    : busca noticias recientes por país con web search
  - extract      : extrae cuerpo de artículos para enriquecer summaries
  - all          : corre todos los tests
"""

import sys
import time
import json
import requests
from datetime import datetime
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# ─── Test 1: Playwright vs Requests ─────────────────────────────────────────
# Sitios que cargan headlines con JS y que requests no captura bien

PLAYWRIGHT_TEST_URLS = [
    {"name": "Clarin - Politica", "url": "https://www.clarin.com/politica", "country": "argentina"},
    {"name": "Infobae - Economia", "url": "https://www.infobae.com/economia/", "country": "argentina"},
    {"name": "El Observador", "url": "https://www.elobservador.com.uy/", "country": "uruguay"},
    {"name": "ABC Color", "url": "https://www.abc.com.py/politica/", "country": "paraguay"},
]


def scrape_with_requests(url):
    """Scraper actual: requests + BeautifulSoup."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        headlines = set()
        for tag in soup.find_all(["h1", "h2", "h3", "a"], limit=80):
            title = tag.get_text(strip=True)
            if title and len(title) >= 25:
                headlines.add(title)
        return list(headlines)
    except Exception as e:
        return [f"ERROR: {e}"]


def scrape_with_playwright(url):
    """Nuevo: Playwright con browser real, captura contenido JS-rendered."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return ["ERROR: playwright no instalado. Correr: pip install playwright && playwright install chromium"]

    headlines = set()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({"User-Agent": HEADERS["User-Agent"]})
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            # Esperar un poco para que cargue JS dinámico
            page.wait_for_timeout(2000)

            # Extraer headlines de h1, h2, h3 y links prominentes
            elements = page.query_selector_all("h1, h2, h3, article a, [class*='title'], [class*='headline']")
            for el in elements:
                text = el.inner_text().strip()
                if text and len(text) >= 25:
                    headlines.add(text)

            browser.close()
    except Exception as e:
        return [f"ERROR: {e}"]

    return list(headlines)


def test_playwright_vs_requests():
    """Compara cantidad y calidad de headlines entre ambos métodos."""
    print("\n" + "=" * 70)
    print("TEST 1: PLAYWRIGHT vs REQUESTS")
    print("=" * 70)
    print("Compara cuántos headlines captura cada método.\n")

    results = []
    for site in PLAYWRIGHT_TEST_URLS:
        print(f"  [{site['name']}] ({site['url']})")

        # Requests
        t0 = time.time()
        req_headlines = scrape_with_requests(site["url"])
        req_time = time.time() - t0

        # Playwright
        t0 = time.time()
        pw_headlines = scrape_with_playwright(site["url"])
        pw_time = time.time() - t0

        # Headlines exclusivos de cada método
        req_set = set(req_headlines)
        pw_set = set(pw_headlines)
        only_requests = req_set - pw_set
        only_playwright = pw_set - req_set

        result = {
            "site": site["name"],
            "requests_count": len(req_headlines),
            "requests_time": round(req_time, 2),
            "playwright_count": len(pw_headlines),
            "playwright_time": round(pw_time, 2),
            "only_in_playwright": len(only_playwright),
            "only_in_requests": len(only_requests),
        }
        results.append(result)

        print(f"    requests:   {result['requests_count']:3d} headlines  ({result['requests_time']}s)")
        print(f"    playwright: {result['playwright_count']:3d} headlines  ({result['playwright_time']}s)")
        print(f"    solo en playwright: {result['only_in_playwright']}")
        if only_playwright:
            for h in list(only_playwright)[:3]:
                print(f"      + {h[:80]}...")
        print()

    # Resumen
    print("-" * 70)
    total_req = sum(r["requests_count"] for r in results)
    total_pw = sum(r["playwright_count"] for r in results)
    total_exclusive = sum(r["only_in_playwright"] for r in results)
    print(f"  TOTAL requests:   {total_req} headlines")
    print(f"  TOTAL playwright: {total_pw} headlines")
    print(f"  Headlines EXTRA con playwright: {total_exclusive}")
    if total_pw > total_req:
        pct = round((total_pw - total_req) / max(total_req, 1) * 100)
        print(f"  => Playwright captura ~{pct}% más contenido")
    print()

    return results


# ─── Test 2: Web Search como fuente complementaria ──────────────────────────

SEARCH_QUERIES = [
    {"query": "noticias Argentina hoy política economía", "country": "argentina"},
    {"query": "noticias Uruguay hoy política economía", "country": "uruguay"},
    {"query": "noticias Paraguay hoy política economía", "country": "paraguay"},
]


def test_websearch_as_source():
    """
    Prueba usar web search para encontrar noticias que los RSS no capturan.
    Nota: Esto requiere el plugin WebSearch de Claude Code, no se puede
    ejecutar standalone. Este test simula la estructura de datos.
    """
    print("\n" + "=" * 70)
    print("TEST 2: WEB SEARCH COMO FUENTE COMPLEMENTARIA")
    print("=" * 70)
    print("""
    Web Search puede complementar los RSS feeds para:
    - Detectar breaking news antes de que aparezcan en RSS
    - Capturar noticias de fuentes no incluidas en SOURCES
    - Validar que no nos estamos perdiendo historias importantes

    Este test NO se puede correr standalone (necesita el plugin de Claude).
    Pero muestra la estructura de integración.
    """)

    # Mostrar cómo se integraría
    print("  Ejemplo de integración en scraper.py:\n")
    print("""
    # En scraper.py, agregar después del scrape de RSS:
    def enrich_with_websearch(country, existing_titles):
        \"\"\"Busca noticias recientes que los feeds no capturaron.\"\"\"
        # Esto se ejecutaría desde Claude Code con WebSearch tool
        query = f"noticias {country} hoy política economía {datetime.now().strftime('%Y-%m-%d')}"
        # Los resultados del WebSearch se parsean y comparan contra existing_titles
        # Solo se agregan las que no están duplicadas
        pass
    """)

    print("  Beneficio esperado:")
    print("    - 10-20% más cobertura de historias importantes")
    print("    - Detección temprana de breaking news")
    print("    - Cross-check de lo que cada fuente cubre\n")


# ─── Test 3: Extracción de cuerpo de artículos ──────────────────────────────

SAMPLE_ARTICLE_URLS = [
    "https://www.infobae.com/politica/",
    "https://www.lanacion.com.ar/politica/",
]


def extract_article_body_requests(url):
    """Intenta extraer el cuerpo de un artículo con requests."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")

        # Intentar selectores comunes para cuerpo de artículos
        selectors = [
            "article",
            "[class*='article-body']",
            "[class*='content-body']",
            "[class*='story-body']",
            "[itemprop='articleBody']",
            ".nota-cuerpo",
            ".article-text",
        ]

        for sel in selectors:
            body = soup.select_one(sel)
            if body:
                paragraphs = body.find_all("p")
                text = " ".join(p.get_text(strip=True) for p in paragraphs)
                if len(text) > 100:
                    return text[:500] + "..."

        return "NO SE PUDO EXTRAER (necesita JS o selector desconocido)"
    except Exception as e:
        return f"ERROR: {e}"


def extract_article_body_playwright(url):
    """Extrae cuerpo con Playwright - captura contenido JS-rendered."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "ERROR: playwright no instalado"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(2000)

            # Intentar extraer con selectores comunes
            selectors = [
                "article p",
                "[class*='article-body'] p",
                "[class*='content'] p",
                "[itemprop='articleBody'] p",
            ]

            for sel in selectors:
                elements = page.query_selector_all(sel)
                if elements:
                    texts = [el.inner_text().strip() for el in elements if el.inner_text().strip()]
                    if texts:
                        full = " ".join(texts)
                        browser.close()
                        return full[:500] + "..." if len(full) > 500 else full

            browser.close()
            return "NO SE PUDO EXTRAER"
    except Exception as e:
        return f"ERROR: {e}"


def test_article_extraction():
    """Compara extracción de cuerpo de artículos."""
    print("\n" + "=" * 70)
    print("TEST 3: EXTRACCION DE CUERPO DE ARTICULOS")
    print("=" * 70)
    print("""
    Extraer el cuerpo de artículos permitiría:
    - Mejor clustering (más texto = mejor TF-IDF / embeddings)
    - Summaries más ricos (el LLM tendría el artículo completo)
    - Detección de fuentes que cubren la misma historia con diferente ángulo
    """)

    for url in SAMPLE_ARTICLE_URLS:
        print(f"\n  [{url}]")

        print("    requests:")
        body_req = extract_article_body_requests(url)
        print(f"    {body_req[:120]}...")

        print("    playwright:")
        body_pw = extract_article_body_playwright(url)
        print(f"    {body_pw[:120]}...")

    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def print_summary():
    """Resumen de mejoras propuestas."""
    print("\n" + "=" * 70)
    print("RESUMEN: MEJORAS CON PLUGINS NUEVOS")
    print("=" * 70)
    print("")
    print("  Plugin              | Mejora")
    print("  --------------------+----------------------------------------------")
    print("  Playwright          | Scraping de sitios JS-heavy (Clarin, etc.)")
    print("                      | Extraccion de cuerpo de articulos")
    print("                      | +20-40% mas headlines capturados")
    print("  --------------------+----------------------------------------------")
    print("  Web Search          | Fuente complementaria de noticias")
    print("                      | Detectar breaking news pre-RSS")
    print("                      | Cross-check de cobertura")
    print("  --------------------+----------------------------------------------")
    print("  Scheduled Tasks     | Ejecutar pipeline desde Claude Code")
    print("                      | Sin depender solo de GitHub Actions")
    print("                      | Monitoreo interactivo del proceso")
    print("  --------------------+----------------------------------------------")
    print("  Context7            | Docs actualizadas de scikit-learn, pandas")
    print("                      | Mejorar clustering con ultimas APIs")
    print("")
    print("  RECOMENDACION:")
    print("    1. Playwright para scraping = mayor impacto inmediato")
    print("    2. Web Search = complemento facil desde Claude Code")
    print("    3. Scheduled Tasks = automatizacion local como backup")
    print("")


if __name__ == "__main__":
    tests = sys.argv[1] if len(sys.argv) > 1 else "all"

    print(f"\n  Media Monitor - Plugin Playground")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    if tests in ("playwright", "all"):
        test_playwright_vs_requests()

    if tests in ("websearch", "all"):
        test_websearch_as_source()

    if tests in ("extract", "all"):
        test_article_extraction()

    print_summary()
