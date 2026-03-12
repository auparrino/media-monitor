"""50 realistic demo articles for fallback when scraping fails."""

import os
import sqlite3
from datetime import datetime, timedelta
import random

# (source, country, title, category)
DEMO_ARTICLES = [
    # Argentina — Politica
    ("Infobae", "argentina", "Milei convocó a sesiones extraordinarias para tratar reforma laboral", "politica"),
    ("La Nacion", "argentina", "Congreso debatirá reforma del sistema jubilatorio en marzo", "politica"),
    ("Clarin", "argentina", "Oposición presentó proyecto alternativo de reforma laboral en el Senado", "politica"),
    ("Infobae", "argentina", "Gobernadores del norte firmaron acuerdo de cooperación con el gobierno nacional", "politica"),
    ("La Nacion", "argentina", "Gabinete nacional incorporó dos nuevos secretarios en áreas estratégicas", "politica"),
    ("Clarin", "argentina", "Encuesta: aprobación del gobierno se mantiene estable en 45%", "politica"),
    ("Infobae", "argentina", "Senado aprobó en general la ley de reforma del Estado", "politica"),

    # Argentina — Economia
    ("Infobae", "argentina", "Reservas del BCRA superaron los USD 30.000 millones por primera vez en dos años", "economia"),
    ("La Nacion", "argentina", "El tipo de cambio oficial cerró estable tras el anuncio del FMI", "economia"),
    ("Ambito", "argentina", "Inflación de febrero se ubicó en 2,4% según el INDEC", "economia"),
    ("Infobae", "argentina", "Vaca Muerta alcanzó récord de producción de gas natural en el primer bimestre", "economia"),
    ("La Nacion", "argentina", "El riesgo país cayó por debajo de los 700 puntos por primera vez desde 2019", "economia"),
    ("Ambito", "argentina", "YPF firmó acuerdo con empresa australiana para exploración en Vaca Muerta", "economia"),
    ("Clarin", "argentina", "Bonos argentinos subieron 3% tras señales positivas del FMI sobre desembolso", "economia"),
    ("Ambito", "argentina", "Tarifas de energía tendrán nuevo ajuste del 15% a partir de abril", "economia"),
    ("La Nacion", "argentina", "Dólar blue se mantuvo estable en torno a los $1.200 durante la semana", "economia"),
    ("Infobae", "argentina", "Exportaciones de carne vacuna crecieron 22% interanual en febrero", "economia"),

    # Argentina — Internacional
    ("Infobae", "argentina", "Cancillería argentina recibió delegación australiana para tratar cooperación en litio", "internacional"),
    ("La Nacion", "argentina", "Argentina y China firmaron memorándum de entendimiento para infraestructura", "internacional"),
    ("Clarin", "argentina", "Milei viajará a Estados Unidos para reunirse con funcionarios del Tesoro", "internacional"),
    ("Infobae", "argentina", "Mercosur-UE: negociaciones entran en etapa de ratificación parlamentaria", "internacional"),
    ("La Nacion", "argentina", "Argentina se posiciona como segundo exportador mundial de litio", "internacional"),

    # Uruguay — Politica
    ("El Observador", "uruguay", "Presidente Orsi anunció reforma del sistema de salud pública", "politica"),
    ("El Pais UY", "uruguay", "Frente Amplio presentó proyecto de ley para vivienda social", "politica"),
    ("El Observador", "uruguay", "Parlamento uruguayo aprobó nueva ley de medios de comunicación", "politica"),
    ("El Pais UY", "uruguay", "Gobierno de Orsi cumplió primeros 100 días con aprobación del 52%", "politica"),
    ("El Observador", "uruguay", "Senado debatirá proyecto de seguridad ciudadana la próxima semana", "politica"),

    # Uruguay — Economia
    ("El Observador", "uruguay", "Banco Central de Uruguay mantuvo tasa de referencia sin cambios", "economia"),
    ("El Pais UY", "uruguay", "Inflación en Uruguay cerró febrero en 0,4% mensual", "economia"),
    ("El Observador", "uruguay", "Sector tecnológico uruguayo creció 18% en exportaciones de servicios", "economia"),
    ("El Pais UY", "uruguay", "Déficit fiscal de Uruguay se redujo al 2,8% del PBI", "economia"),
    ("El Observador", "uruguay", "Exportaciones de celulosa marcaron récord en enero con USD 280 millones", "economia"),

    # Uruguay — Internacional
    ("El Observador", "uruguay", "Uruguay y Australia avanzaron en acuerdo de cooperación en minería", "internacional"),
    ("El Pais UY", "uruguay", "Canciller uruguayo visitó Bruselas para acelerar acuerdo Mercosur-UE", "internacional"),
    ("El Observador", "uruguay", "Uruguay negocia acceso preferencial al mercado asiático para carne", "internacional"),

    # Paraguay — Politica
    ("ABC Color", "paraguay", "Presidente Santiago Peña se reunió con inversores del sector minero", "politica"),
    ("Ultima Hora", "paraguay", "Senado paraguayo debatirá reforma electoral en sesión extraordinaria", "politica"),
    ("ABC Color", "paraguay", "Gobierno paraguayo lanzó plan de modernización del Estado", "politica"),
    ("Ultima Hora", "paraguay", "Diputados aprobaron ley de transparencia para funcionarios públicos", "politica"),
    ("ABC Color", "paraguay", "Partido Colorado consolidó alianza con sectores independientes", "politica"),

    # Paraguay — Economia
    ("ABC Color", "paraguay", "Paraguay registró récord de exportaciones de soja en febrero", "economia"),
    ("Ultima Hora", "paraguay", "Banco Central de Paraguay redujo tasa de interés a 5,5%", "economia"),
    ("ABC Color", "paraguay", "Inversión extranjera directa en Paraguay creció 30% en 2025", "economia"),
    ("Ultima Hora", "paraguay", "Presupuesto nacional 2026 prioriza infraestructura vial y educación", "economia"),
    ("ABC Color", "paraguay", "Exportaciones paraguayas de carne superaron USD 200 millones en enero", "economia"),

    # Paraguay — Internacional
    ("ABC Color", "paraguay", "Paraguay y EEUU firmaron acuerdo de cooperación en seguridad fronteriza", "internacional"),
    ("Ultima Hora", "paraguay", "Cancillería paraguaya abrirá nueva embajada en Australia", "internacional"),
    ("Ultima Hora", "paraguay", "Paraguay busca ampliar mercados para maíz en Europa y Asia", "internacional"),
    ("ABC Color", "paraguay", "Cumbre del Mercosur en Asunción definirá agenda comercial 2026", "internacional"),

    # Extra
    ("El Observador", "uruguay", "Uruguay apuesta por energías renovables con nuevo parque eólico", "economia"),
]


def load_demo_data(db_path=None):
    """Insert demo articles into the database."""
    if db_path is None:
        db_path = os.path.join(os.path.dirname(__file__), "data", "news.db")

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
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

    now = datetime.now()
    scraped_at = now.isoformat()

    for i, (source, country, title, category) in enumerate(DEMO_ARTICLES):
        days_ago = i % 7
        hours_offset = random.randint(0, 23)
        pub_date = (now - timedelta(days=days_ago, hours=hours_offset)).isoformat()
        slug = title.lower().replace(" ", "-")[:60]
        url = f"https://demo.example.com/{country}/{slug}-{i}"

        try:
            conn.execute(
                "INSERT OR IGNORE INTO articles (source, country, title, url, published_at, scraped_at, category) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (source, country, title, url, pub_date, scraped_at, category),
            )
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    print(f"   {len(DEMO_ARTICLES)} articulos demo cargados")


if __name__ == "__main__":
    load_demo_data()
