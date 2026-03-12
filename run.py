#!/usr/bin/env python3
"""Master script — run scraper then generate dashboard."""

import sys
import os
from datetime import datetime

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Ensure we're running from the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    print(f"\n{'='*50}")
    print(f"  CONO SUR MEDIA MONITOR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}\n")

    print("▶  [1/2] Scraping noticias...")
    from scraper import run_scraper
    run_scraper()

    print("\n▶  [2/2] Generando dashboard...")
    from dashboard import generate_dashboard
    generate_dashboard()

    print(f"\n✓  Listo: output/dashboard.html")
    print(f"   Abrí en browser e imprimí con Ctrl+P → A4\n")


if __name__ == "__main__":
    main()
