import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as Wait
from webdriver_manager.chrome import ChromeDriverManager

from flashscore_scraper.competition import CompSeason
from flashscore_scraper.match import Match
from flashscore_scraper.utils import hide_sdk_banner

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
    hide_sdk_banner(driver)

    c.load_comp_season_match_page(driver)

    list_of_matches = []

    # <loop through all the relevant matches>
    Wait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".soccer .event__match--static")))
    matches = driver.find_elements(By.CSS_SELECTOR, ".soccer .event__match--static")

    try:
        wizard_element = driver.find_element(By.CSS_SELECTOR, ".wizard")
        driver.execute_script("arguments[0].style.display = 'none';", wizard_element)
        time.sleep(2)
    except NoSuchElementException:
        pass

    for match in matches:
        match.click()
        time.sleep(2)

        new_window = driver.window_handles[1]
        driver.switch_to.window(new_window)
        hide_sdk_banner(driver, sleep=2.5, include_placeholder=False)

        Wait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//a[@href='/football/" + c.country2 + "/" + c.name2 + "/']"))
        )
        competition_stage = driver.find_element(By.XPATH, "//a[@href='/football/" + c.country2 + "/" + c.name2 + "/']")
        if re.match(r"" + c.name1.upper() + " - ((APERTURA|CLAUSURA) - )?ROUND.*", competition_stage.text):
            print(driver.title)

            new_match = Match()
            new_match.get_match_statistics(driver, c.country1, c.name1, c.season)

            if new_match.match_valid:
                list_of_matches.append(new_match)

        driver.close()
        driver.switch_to.window(driver.window_handles[0])

    Match.correct_zero_values(list_of_matches)
    # Match.check_num_of_matches(list_of_matches, c)  TODO: Uncomment?
    print(f"{len(list_of_matches)} matches were found.\n")

    new_df = pd.DataFrame([match.to_dict() for match in list_of_matches])
    df = pd.concat([df, new_df], ignore_index=True)

    driver.quit()

# final dedupe over the accumulated dataframe and overwrite cleanly
if not df.empty:
    df = Match.drop_duplicate_matches(df)
    df.to_csv(CSV, index=False)

break_point = 0
