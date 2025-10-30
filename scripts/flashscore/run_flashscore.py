import json
import os
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as Wait
from webdriver_manager.chrome import ChromeDriverManager

from flashscore_scraper.competition import CompSeason
from flashscore_scraper.match import Match
from flashscore_scraper.utils import dismiss_cookie_banner, hide_sdk_banner

DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV = DATA_DIR / "flashscore.matches.csv"


def load_existing_matches() -> pd.DataFrame:
    """Return existing matches or an empty DataFrame if the file is missing/empty."""
    if not CSV.exists() or os.stat(CSV).st_size == 0:
        # no file yet or empty file -> start fresh
        return pd.DataFrame()

    try:
        return pd.read_csv(CSV)
    except EmptyDataError:
        # file exists but has no rows/headers
        return pd.DataFrame()


def save_matches(df: pd.DataFrame) -> None:
    """Append a chunk of matches to the CSV (write header only once)."""
    mode = "w" if not CSV.exists() or os.stat(CSV).st_size == 0 else "a"
    header = mode == "w"
    df.to_csv(CSV, index=False, mode=mode, header=header)


def open_stats_tab(driver, timeout=6):
    """Navigate to the Stats sub-tab on the match page."""
    # 1) Try data attribute (stable)
    for sel in [
        "a[data-analytics-alias='match-statistics']",
        "a[href*='/summary/stats']",
        "button[role='tab']:scope",  # fallback: search by visible text
    ]:
        try:
            if sel == "button[role='tab']:scope":
                tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
                for t in tabs:
                    try:
                        if "stats" in t.text.strip().lower():
                            driver.execute_script("arguments[0].click();", t)
                            break
                    except Exception:
                        continue
            else:
                el = driver.find_element(By.CSS_SELECTOR, sel)
                driver.execute_script("arguments[0].click();", el)
            break
        except NoSuchElementException:
            continue

    # 2) Wait for stats widget to appear (don’t hang forever)
    try:
        Wait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='wcl-statistics']")))
    except TimeoutException:
        # Stats widget did not load; continue anyway (your parser handles None gracefully)
        pass


# Read settings.json file
HERE = Path(__file__).resolve().parent
with open(HERE / "comp_settings.json", "r", encoding="utf-8") as f:
    comp_settings = json.load(f)

# Create CompSeason instances
comp_seasons = []
for s in comp_settings:
    comp = CompSeason()
    comp.__dict__.update(s)
    comp_seasons.append(comp)

# Start from whatever is already in the CSV (safe if empty / missing)
df = load_existing_matches()

# Scrape
for c in comp_seasons:
    # Set webdriver
    options = Options()
    options.add_experimental_option("detach", True)
    driver: WebDriver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # Load main page
    driver.get("https://www.flashscore.com/")
    driver.maximize_window()
    dismiss_cookie_banner(driver)
    hide_sdk_banner(driver)

    c.load_comp_season_match_page(driver)

    # Flashscore results rows
    rows = driver.find_elements(By.CSS_SELECTOR, "div.event__match")
    match_urls = []
    for r in rows:
        try:
            a = r.find_element(By.CSS_SELECTOR, "a.eventRowLink")
            href = a.get_attribute("href")
            if href:
                match_urls.append(href)
        except NoSuchElementException:
            continue

    print(f"Collected {len(match_urls)} match URLs.")

    scraped_matches = []

    # <loop through all the relevant matches>
    Wait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".soccer .event__match--static")))
    # matches = driver.find_elements(By.CSS_SELECTOR, ".soccer .event__match--static")

    # --- Iterate by navigating directly to each URL (no stale elements) ---
    for i, url in enumerate(match_urls, 1):
        try:
            driver.get(url)
            dismiss_cookie_banner(driver)  # will be a no-op after first time

            m = Match()
            m.get_match_statistics(driver, c.country1, c.name1, c.season)

            if not m.match_valid:
                print(f"⚠️ Match {i} marked invalid, skipping save.")
                continue

            # append immediately so you don't lose progress if you stop the run
            # row_df = pd.DataFrame([m.to_dict()])
            # build row dataframe
            payload = m.to_dict()
            row_df = pd.DataFrame([payload])

            # enforce canonical column order (prevents legacy columns)
            cols = list(payload.keys())
            row_df = row_df.reindex(columns=cols)

            mode = "a" if CSV.exists() and CSV.stat().st_size > 0 else "w"
            row_df.to_csv(CSV, index=False, mode=mode, header=(mode == "w"))
            scraped_matches.append(m)
            print(f"✅ Match {i} appended to CSV")

        except Exception as e:
            print(f"⚠️ Match {i} failed: {e}")
            continue

    Match.correct_zero_values(scraped_matches)
    # Match.check_num_of_matches(list_of_matches, c)  TODO: Possibly uncomment
    print(f"{len(scraped_matches)} matches were found.\n")

    driver.quit()

# TODO: Check for duplicate matches?
"""
# final dedupe over the accumulated dataframe and overwrite cleanly
if not df.empty:
    df = Match.drop_duplicate_matches(df)
    df.to_csv(CSV, index=False)
"""
