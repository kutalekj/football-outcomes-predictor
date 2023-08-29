import time
import re
import pandas as pd
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as Wait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

from match import Match


def hide_sdk_banner(sleep=2.0, include_placeholder=True):
    time.sleep(sleep)
    driver.execute_script("document.getElementById('onetrust-banner-sdk').style.display='none';")
    if include_placeholder:
        driver.execute_script("document.getElementsByClassName('otPlaceholder')[0].style.display='none';")


def hide_advert_banner(sleep=2):
    time.sleep(sleep)
    driver.execute_script(
        "document.getElementsByClassName('boxOverContent boxOverContent--type-2 isSticky isMobileSticky disabledSkeleton isNotClosed boxOverContent--active')[0].style.display='none';")


def load_fortuna_liga_match_page():
    # Show more countries -> Czech Republic -> FORTUNA:LIGA
    time.sleep(1.5)
    Wait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "lmc__itemMore")))
    driver.find_element(By.CLASS_NAME, "lmc__itemMore").click()

    Wait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//span[text()='Czech Republic']")))
    driver.find_element(By.XPATH, "//span[text()='Czech Republic']").click()

    Wait(driver, 10).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'a[href="/football/czech-republic/fortuna-liga/"].lmc__templateHref')))
    driver.find_element(By.CSS_SELECTOR, 'a[href="/football/czech-republic/fortuna-liga/"].lmc__templateHref').click()

    # Archive -> FORTUNA:LIGA 2022/2023 -> Results
    hide_sdk_banner()
    Wait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//div[@class="heading__name" and text()="FORTUNA:LIGA"]')))
    driver.find_element(By.CSS_SELECTOR,
                        'a[href="/football/czech-republic/fortuna-liga/archive/"]#li5.tabs__tab.archive').click()

    hide_sdk_banner()
    Wait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR,
                                                           'a.archive__text.archive__text--clickable[href="/football/czech-republic/fortuna-liga-2022-2023/"]')))
    driver.find_element(By.CSS_SELECTOR,
                        'a.archive__text.archive__text--clickable[href="/football/czech-republic/fortuna-liga-2022-2023/"]').click()

    hide_sdk_banner()
    Wait(driver, 10).until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, 'a[href="/football/czech-republic/fortuna-liga-2022-2023/results/"]#li2.tabs__tab.results')))
    driver.find_element(By.CSS_SELECTOR,
                        'a[href="/football/czech-republic/fortuna-liga-2022-2023/results/"]#li2.tabs__tab.results').click()

    # Show more matches -> Show more matches... (max. 3 times)
    hide_sdk_banner()
    Wait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'a.event__more.event__more--static[href="#"]')))
    try:
        driver.find_element(By.CSS_SELECTOR, 'a.event__more.event__more--static[href="#"]').click()
        try:
            Wait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a.event__more.event__more--static[href="#"]')))
            hide_advert_banner()
            driver.find_element(By.CSS_SELECTOR, 'a.event__more.event__more--static[href="#"]').click()
            try:
                Wait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'a.event__more.event__more--static[href="#"]')))
                hide_advert_banner()
                driver.find_element(By.CSS_SELECTOR, 'a.event__more.event__more--static[href="#"]').click()
            except NoSuchElementException:
                print("Warning: Element not present")

        except NoSuchElementException:
            print("Warning: Element not present")

    except NoSuchElementException:
        print("Warning: Element not present")

    finally:
        print("All matches found successfully. Continuing...")


# Set webdriver
options = Options()
options.add_experimental_option("detach", True)
driver: WebDriver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Load main page
driver.get("https://www.flashscore.com/")
driver.maximize_window()
hide_sdk_banner()

# ------------------------------------------------------------------- FORTUNA:LIGA 2022/2023
list_of_matches = []

load_fortuna_liga_match_page()

# <loop through all the relevant matches>
matches = driver.find_elements(By.CSS_SELECTOR, '.soccer .event__match--static')
for match in matches[106:110]:
    match.click()
    time.sleep(2)

    new_window = driver.window_handles[1]
    driver.switch_to.window(new_window)
    hide_sdk_banner(include_placeholder=False, sleep=1.5)

    Wait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//a[@href='/football/czech-republic/fortuna-liga/']")))
    competition_stage = driver.find_element(By.XPATH, "//a[@href='/football/czech-republic/fortuna-liga/']")
    if re.match(r'FORTUNA:LIGA - ROUND.*', competition_stage.text):
        print("\n" + driver.title)

        # TODO: Get match statistics
        new_match = Match()
        new_match.get_match_statistics(driver, "FORTUNA:LIGA", "2022-2023")
        list_of_matches.append(new_match)

    driver.close()
    driver.switch_to.window(driver.window_handles[0])

df = pd.DataFrame([match.to_dict() for match in list_of_matches])

break_point = 0
