import time
import re
import pandas as pd

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


# Set webdriver
options = Options()
options.add_experimental_option("detach", True)
driver: WebDriver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Load main page
driver.get("https://www.flashscore.com/")
driver.maximize_window()
hide_sdk_banner(driver)

# ------------------------------------------------------------------- FORTUNA:LIGA 2022/2023
comp = CompSeason()
comp.country1 = "Czech Republic"
comp.country2 = "czech-republic"
comp.name1 = "FORTUNA:LIGA"
comp.name2 = "fortuna-liga"
comp.season = "2021-2022"

comp.load_comp_season_match_page(driver)

list_of_matches = []

# <loop through all the relevant matches>
matches = driver.find_elements(By.CSS_SELECTOR, '.soccer .event__match--static')
for match in matches[100:210]:
    match.click()
    time.sleep(2)

    new_window = driver.window_handles[1]
    driver.switch_to.window(new_window)
    hide_sdk_banner(driver, include_placeholder=False, sleep=1.5)

    Wait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//a[@href='/football/" + comp.country2 + '/' + comp.name2 + "/']")))
    competition_stage = driver.find_element(By.XPATH, "//a[@href='/football/" + comp.country2 + '/' + comp.name2 + "/']")
    if re.match(r'' + comp.name1 + ' - ROUND.*', competition_stage.text):
        print("\n" + driver.title)

        new_match = Match()
        new_match.get_match_statistics(driver, comp.name1, comp.season)
        list_of_matches.append(new_match)

    driver.close()
    driver.switch_to.window(driver.window_handles[0])

# Possible correction of -1 values to 0
Match.correct_zero_values(list_of_matches)

df = pd.DataFrame([match.to_dict() for match in list_of_matches])

df.to_csv('matches.csv', index=False)

break_point = 0
