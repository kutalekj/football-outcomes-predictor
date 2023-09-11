import time
import re
import pandas as pd
import json
import os

from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as Wait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from match import Match
from competition import CompSeason
from utils import hide_sdk_banner

file_path = "matches.csv"

df = pd.DataFrame()

if os.path.isfile(file_path):
    df = pd.read_csv(file_path)
else:
    print(f"Could not find and open the file {file_path}.")

# Read settings.json file
with open('../comp_settings.json', 'r') as f:
    comp_settings = json.load(f)

# Create CompSeason instances
comp_seasons = []
for s in comp_settings:
    comp = CompSeason()
    comp.__dict__.update(s)
    comp_seasons.append(comp)

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
    Wait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.soccer .event__match--static')))
    matches = driver.find_elements(By.CSS_SELECTOR, '.soccer .event__match--static')
    for match in matches:
        match.click()
        time.sleep(2)

        new_window = driver.window_handles[1]
        driver.switch_to.window(new_window)
        hide_sdk_banner(driver, include_placeholder=False, sleep=1.5)

        Wait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//a[@href='/football/" + c.country2 + '/' + c.name2 + "/']")))
        competition_stage = driver.find_element(By.XPATH, "//a[@href='/football/" + c.country2 + '/' + c.name2 + "/']")
        if re.match(r'' + c.name1.upper() + ' - ROUND.*', competition_stage.text):
            print(driver.title)

            new_match = Match()
            new_match.get_match_statistics(driver, c.country1, c.name1, c.season)

            if new_match.match_valid:
                list_of_matches.append(new_match)

        driver.close()
        driver.switch_to.window(driver.window_handles[0])

    Match.correct_zero_values(list_of_matches)
    # Match.check_num_of_matches(list_of_matches, c)  TODO: Uncomment
    print(f"{len(list_of_matches)} matches were found.\n")

    new_df = pd.DataFrame([match.to_dict() for match in list_of_matches])
    df = pd.concat([df, new_df], ignore_index=True)

    driver.quit()

df.drop_duplicates(inplace=True)

df.to_csv(file_path, index=False)

break_point = 0
