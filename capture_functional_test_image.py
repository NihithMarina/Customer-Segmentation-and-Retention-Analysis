from pathlib import Path

from playwright.sync_api import sync_playwright


BASE_DIR = Path(__file__).resolve().parent
URL = "http://localhost:8503"
OUTPUT = BASE_DIR.parent / "screenshots" / "report_images" / "functional_testing_dashboard.png"
CSV_TO_UPLOAD = BASE_DIR / "customer_segments.csv"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(URL, wait_until="networkidle", timeout=120000)
        page.wait_for_timeout(4000)

        # Upload a valid CSV using the sidebar uploader.
        page.locator("input[type='file']").set_input_files(str(CSV_TO_UPLOAD))
        page.wait_for_timeout(3500)

        # Apply one filter for test evidence (click visible label text).
        page.get_by_text("Churned only", exact=True).click(timeout=15000)
        page.wait_for_timeout(1500)

        # Open Data Explorer so table output is visible in the screenshot.
        page.get_by_role("tab", name="Data Explorer", exact=True).click(timeout=15000)
        page.wait_for_timeout(2000)

        page.screenshot(path=str(OUTPUT), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
