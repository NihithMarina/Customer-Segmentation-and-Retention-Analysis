from pathlib import Path
import re

from playwright.sync_api import sync_playwright


URL = "http://localhost:8502"
OUTPUT_DIR = Path(r"c:\Users\NIHITH\OneDrive\Documents\Customer Segmentation & Retention Analysis\screenshots")
TAB_NAMES = [
    "Executive Overview",
    "Segment Studio",
    "Cohort and Product Lab",
    "Retention Simulator",
    "Action Console",
    "Data Explorer",
]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(URL, wait_until="networkidle", timeout=120000)

        # Allow Streamlit to complete first render.
        page.wait_for_timeout(4000)

        page.screenshot(path=str(OUTPUT_DIR / "00_dashboard_home.png"), full_page=True)

        for idx, tab_name in enumerate(TAB_NAMES, start=1):
            tab = page.get_by_role("tab", name=tab_name, exact=True)
            tab.click(timeout=15000)
            page.wait_for_timeout(2200)
            file_name = f"{idx:02d}_{slugify(tab_name)}.png"
            page.screenshot(path=str(OUTPUT_DIR / file_name), full_page=True)

        # Bonus mobile layout screenshot.
        mobile = browser.new_page(viewport={"width": 412, "height": 915})
        mobile.goto(URL, wait_until="networkidle", timeout=120000)
        mobile.wait_for_timeout(4000)
        mobile.screenshot(path=str(OUTPUT_DIR / "07_mobile_home.png"), full_page=True)

        browser.close()


if __name__ == "__main__":
    main()
