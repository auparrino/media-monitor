"""RSS scraper with Playwright-enhanced HTML fallback. Category = source section."""

import os
import time
import sqlite3
from datetime import datetime

import requests
import feedparser
from bs4 import BeautifulSoup

# Playwright browser — lazy-loaded, shared across all feeds in a single run
_pw_instance = None
_pw_browser = None


def _get_playwright_browser():
    """Return a shared Playwright browser, launching it on first call."""
    global _pw_instance, _pw_browser
    if _pw_browser is not None:
        return _pw_browser
    try:
        from playwright.sync_api import sync_playwright
        _pw_instance = sync_playwright().start()
        _pw_browser = _pw_instance.chromium.launch(headless=True)
        return _pw_browser
    except Exception:
        return None


def _close_playwright():
    """Shut down the shared Playwright browser if it was started."""
    global _pw_instance, _pw_browser
    if _pw_browser:
        try:
            _pw_browser.close()
        except Exception:
            pass
        _pw_browser = None
    if _pw_instance:
        try:
            _pw_instance.stop()
        except Exception:
            pass
        _pw_instance = None

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "news.db")

# Each feed has an explicit section → category mapping
# "type": "rss" = try RSS first; "html" = HTML scrape only (used as fallback)
SOURCES = [
    # Argentina — Infobae
    {"name": "Infobae", "country": "argentina", "feeds": [
        {"url": "https://www.infobae.com/arc/outboundfeeds/rss/category/politica/", "category": "politica", "type": "rss"},
        {"url": "https://www.infobae.com/arc/outboundfeeds/rss/category/economia/", "category": "economia", "type": "rss"},
        {"url": "https://www.infobae.com/arc/outboundfeeds/rss/category/el-mundo/", "category": "internacional", "type": "rss"},
        {"url": "https://www.infobae.com/politica/", "category": "politica", "type": "html"},
        {"url": "https://www.infobae.com/economia/", "category": "economia", "type": "html"},
    ]},
    # Argentina — La Nacion
    {"name": "La Nacion", "country": "argentina", "feeds": [
        {"url": "https://www.lanacion.com.ar/arcio/rss/category/politica/", "category": "politica", "type": "rss"},
        {"url": "https://www.lanacion.com.ar/arcio/rss/category/economia/", "category": "economia", "type": "rss"},
        {"url": "https://www.lanacion.com.ar/arcio/rss/category/el-mundo/", "category": "internacional", "type": "rss"},
        {"url": "https://www.lanacion.com.ar/politica/", "category": "politica", "type": "html"},
        {"url": "https://www.lanacion.com.ar/economia/", "category": "economia", "type": "html"},
    ]},
    # Argentina — Clarin
    {"name": "Clarin", "country": "argentina", "feeds": [
        {"url": "https://www.clarin.com/rss/politica/", "category": "politica", "type": "rss"},
        {"url": "https://www.clarin.com/rss/economia/", "category": "economia", "type": "rss"},
        {"url": "https://www.clarin.com/rss/mundo/", "category": "internacional", "type": "rss"},
        {"url": "https://www.clarin.com/politica", "category": "politica", "type": "html"},
        {"url": "https://www.clarin.com/economia", "category": "economia", "type": "html"},
        {"url": "https://www.clarin.com/mundo", "category": "internacional", "type": "html"},
    ]},
    # Argentina — Ambito
    {"name": "Ambito", "country": "argentina", "feeds": [
        {"url": "https://www.ambito.com/rss/pages/home.xml", "category": "economia", "type": "rss"},
        {"url": "https://www.ambito.com/politica", "category": "politica", "type": "html"},
        {"url": "https://www.ambito.com/economia", "category": "economia", "type": "html"},
    ]},
    # Uruguay — El Observador
    {"name": "El Observador", "country": "uruguay", "feeds": [
        {"url": "https://www.elobservador.com.uy/rss/pages/nacional.xml", "category": "politica", "type": "rss"},
        {"url": "https://www.elobservador.com.uy/rss/pages/economia.xml", "category": "economia", "type": "rss"},
        {"url": "https://www.elobservador.com.uy/rss/pages/Mundo.xml", "category": "internacional", "type": "rss"},
        {"url": "https://www.elobservador.com.uy/seccion/economia", "category": "economia", "type": "html"},
        {"url": "https://www.elobservador.com.uy/seccion/nacional", "category": "politica", "type": "html"},
    ]},
    # Uruguay — El Pais UY (no section RSS, HTML only)
    {"name": "El Pais UY", "country": "uruguay", "feeds": [
        {"url": "https://www.elpais.com.uy/informacion/politica", "category": "politica", "type": "html"},
        {"url": "https://www.elpais.com.uy/economia", "category": "economia", "type": "html"},
        {"url": "https://www.elpais.com.uy/mundo", "category": "internacional", "type": "html"},
    ]},
    # Paraguay — ABC Color
    {"name": "ABC Color", "country": "paraguay", "feeds": [
        {"url": "https://www.abc.com.py/arc/outboundfeeds/rss/nacionales/", "category": "politica", "type": "rss"},
        {"url": "https://www.abc.com.py/arc/outboundfeeds/rss/economia/", "category": "economia", "type": "rss"},
        {"url": "https://www.abc.com.py/arc/outboundfeeds/rss/mundo/", "category": "internacional", "type": "rss"},
        {"url": "https://www.abc.com.py/nacionales/", "category": "politica", "type": "html"},
        {"url": "https://www.abc.com.py/economia/", "category": "economia", "type": "html"},
    ]},
    # Paraguay — Ultima Hora (no RSS)
    {"name": "Ultima Hora", "country": "paraguay", "feeds": [
        {"url": "https://www.ultimahora.com/politica", "category": "politica", "type": "html"},
        {"url": "https://www.ultimahora.com/economia", "category": "economia", "type": "html"},
        {"url": "https://www.ultimahora.com/mundo", "category": "internacional", "type": "html"},
    ]},
    # Argentina — Pagina 12
    {"name": "Pagina 12", "country": "argentina", "feeds": [
        {"url": "https://www.pagina12.com.ar/arc/outboundfeeds/rss/secciones/el-pais/notas", "category": "politica", "type": "rss"},
        {"url": "https://www.pagina12.com.ar/arc/outboundfeeds/rss/secciones/economia/notas", "category": "economia", "type": "rss"},
        {"url": "https://www.pagina12.com.ar/arc/outboundfeeds/rss/secciones/el-mundo/notas", "category": "internacional", "type": "rss"},
        {"url": "https://www.pagina12.com.ar/secciones/el-pais", "category": "politica", "type": "html"},
        {"url": "https://www.pagina12.com.ar/secciones/economia", "category": "economia", "type": "html"},
    ]},
    # Argentina — El Destape (no RSS)
    {"name": "El Destape", "country": "argentina", "feeds": [
        {"url": "https://www.eldestapeweb.com/seccion/politica", "category": "politica", "type": "html"},
        {"url": "https://www.eldestapeweb.com/economia", "category": "economia", "type": "html"},
        {"url": "https://www.eldestapeweb.com/internacionales", "category": "internacional", "type": "html"},
    ]},
    # Uruguay — La Diaria (general RSS includes lifestyle, use section HTML only)
    {"name": "La Diaria", "country": "uruguay", "feeds": [
        {"url": "https://ladiaria.com.uy/politica/", "category": "politica", "type": "html"},
        {"url": "https://ladiaria.com.uy/economia/", "category": "economia", "type": "html"},
        {"url": "https://ladiaria.com.uy/mundo/", "category": "internacional", "type": "html"},
    ]},
    # Uruguay — Montevideo Portal (general feeds include lifestyle, keep only business)
    {"name": "Montevideo Portal", "country": "uruguay", "feeds": [
        {"url": "https://www.montevideo.com.uy/anxml.aspx?728", "category": "economia", "type": "rss"},
        {"url": "https://www.montevideo.com.uy/Noticias/Nacionales-702", "category": "politica", "type": "html"},
    ]},
    # Paraguay — La Nacion PY
    {"name": "La Nacion PY", "country": "paraguay", "feeds": [
        {"url": "https://www.lanacion.com.py/arc/outboundfeeds/rss/category/politica/?outputType=xml", "category": "politica", "type": "rss"},
        {"url": "https://www.lanacion.com.py/arc/outboundfeeds/rss/category/negocios/?outputType=xml", "category": "economia", "type": "rss"},
        {"url": "https://www.lanacion.com.py/arc/outboundfeeds/rss/category/mundo/?outputType=xml", "category": "internacional", "type": "rss"},
        {"url": "https://www.lanacion.com.py/category/politica/", "category": "politica", "type": "html"},
        {"url": "https://www.lanacion.com.py/category/negocios/", "category": "economia", "type": "html"},
    ]},
    # Paraguay — Hoy PY (no RSS)
    {"name": "Hoy", "country": "paraguay", "feeds": [
        {"url": "https://www.hoy.com.py/politica", "category": "politica", "type": "html"},
        {"url": "https://www.hoy.com.py/nacionales", "category": "politica", "type": "html"},
        {"url": "https://www.hoy.com.py/dinero-y-negocios", "category": "economia", "type": "html"},
        {"url": "https://www.hoy.com.py/mundo", "category": "internacional", "type": "html"},
    ]},
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def init_db():
    """Create the database and articles table if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            country TEXT,
            title TEXT,
            url TEXT,
            published_at TEXT,
            scraped_at TEXT,
            category TEXT,
            UNIQUE(url)
        )
    """)
    conn.commit()
    return conn


def _fetch_rss(feed_url, source_name, country, category):
    """Parse a single RSS feed URL. Returns list of article dicts."""
    articles = []
    try:
        resp = requests.get(feed_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        for entry in feed.entries[:25]:
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            pub = entry.get("published", entry.get("updated", ""))
            if title and link:
                articles.append({
                    "source": source_name,
                    "country": country,
                    "title": title,
                    "url": link,
                    "published_at": pub or datetime.now().isoformat(),
                    "category": category,
                })
    except Exception:
        pass
    return articles


def _fetch_html(page_url, source_name, country, category):
    """Scrape headlines from a section page. Returns list of article dicts."""
    articles = []
    try:
        resp = requests.get(page_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        base_url = "/".join(page_url.split("/")[:3])
        seen = set()
        for tag in soup.find_all(["h1", "h2", "h3", "a"], limit=80):
            title = tag.get_text(strip=True)
            link = tag.get("href", "")
            if not title or len(title) < 20 or title in seen:
                continue
            if link and not link.startswith("http"):
                link = base_url.rstrip("/") + "/" + link.lstrip("/")
            if not link:
                continue
            seen.add(title)
            articles.append({
                "source": source_name,
                "country": country,
                "title": title,
                "url": link,
                "published_at": datetime.now().isoformat(),
                "category": category,
            })
    except Exception:
        pass
    return articles


def _fetch_html_playwright(page_url, source_name, country, category):
    """Scrape headlines using Playwright (JS-rendered). Falls back to _fetch_html."""
    browser = _get_playwright_browser()
    if browser is None:
        return _fetch_html(page_url, source_name, country, category)

    articles = []
    try:
        page = browser.new_page()
        page.set_extra_http_headers({"User-Agent": HEADERS["User-Agent"]})
        page.goto(page_url, wait_until="domcontentloaded", timeout=20000)
        page.wait_for_timeout(2000)

        base_url = "/".join(page_url.split("/")[:3])
        seen = set()

        # Broad selector: all anchor tags in article-like containers + headlines
        for el in page.query_selector_all("a"):
            try:
                link = el.get_attribute("href") or ""
                if not link or link == "#" or link.startswith("javascript:"):
                    continue
                title = el.inner_text().strip()
                # Also check if the link wraps a heading
                if not title or len(title) < 20:
                    heading = el.query_selector("h1, h2, h3, h4, span")
                    if heading:
                        title = heading.inner_text().strip()
            except Exception:
                continue
            if not title or len(title) < 20 or title in seen:
                continue
            if not link.startswith("http"):
                link = base_url.rstrip("/") + "/" + link.lstrip("/")
            seen.add(title)
            articles.append({
                "source": source_name,
                "country": country,
                "title": title,
                "url": link,
                "published_at": datetime.now().isoformat(),
                "category": category,
            })

        page.close()
    except Exception:
        # If Playwright fails for this page, fall back to requests
        return _fetch_html(page_url, source_name, country, category)

    # If Playwright got nothing, try requests as last resort
    if not articles:
        return _fetch_html(page_url, source_name, country, category)

    return articles


def scrape_source(source):
    """Scrape all feeds for a source. RSS first, then Playwright HTML to complement."""
    all_articles = []
    seen_urls = set()
    rss_feeds = [f for f in source["feeds"] if f["type"] == "rss"]
    html_feeds = [f for f in source["feeds"] if f["type"] == "html"]

    # Try RSS feeds first
    for feed in rss_feeds:
        arts = _fetch_rss(feed["url"], source["name"], source["country"], feed["category"])
        for a in arts:
            if a["url"] not in seen_urls:
                seen_urls.add(a["url"])
                all_articles.append(a)
        time.sleep(0.5)

    # Always run Playwright HTML to capture JS-rendered headlines RSS may miss
    if html_feeds:
        print("+PW...", end=" ")
        for feed in html_feeds:
            arts = _fetch_html_playwright(feed["url"], source["name"], source["country"], feed["category"])
            for a in arts:
                if a["url"] not in seen_urls:
                    seen_urls.add(a["url"])
                    all_articles.append(a)
            time.sleep(0.5)

    return all_articles


def insert_articles(conn, articles):
    """Insert articles into DB, skipping duplicates."""
    now = datetime.now().isoformat()
    for art in articles:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO articles (source, country, title, url, published_at, scraped_at, category) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (art["source"], art["country"], art["title"], art["url"],
                 art["published_at"], now, art["category"]),
            )
        except sqlite3.IntegrityError:
            pass
    conn.commit()


def run_scraper():
    """Main scraper entry point."""
    conn = init_db()
    total_new = 0
    summary = {}

    for src in SOURCES:
        print(f"   -> {src['name']} ({src['country']})...", end=" ")
        before = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        articles = scrape_source(src)
        insert_articles(conn, articles)
        after = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        added = after - before
        total_new += added
        summary[src["name"]] = added
        print(f"{len(articles)} found, {added} new")
        time.sleep(1)

    print(f"\n   Resumen: {total_new} articulos nuevos en total")
    for name, count in summary.items():
        print(f"     {name}: +{count}")

    # Check if DB has fewer than 15 articles total
    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    if total < 15:
        print(f"\n   Solo {total} articulos en DB. Cargando datos demo...")
        from demo_data import load_demo_data
        load_demo_data(DB_PATH)
        total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        print(f"   DB ahora tiene {total} articulos (con demo data)")

    _close_playwright()
    conn.close()


if __name__ == "__main__":
    run_scraper()
