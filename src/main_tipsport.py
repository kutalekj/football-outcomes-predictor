from utils import hide_tipsport_consent_banner
from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as Wait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import time

# Set webdriver
options = Options()
options.add_experimental_option("detach", True)
# options.add_argument("--headless")
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36")
driver: WebDriver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Load main page
driver.get("https://www.tipsport.cz//")
driver.maximize_window()

# Football
Wait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.Menustyled__FirstLvlTitle-sc-12g27mg-0')))
driver.find_element(By.CSS_SELECTOR, '.Menustyled__FirstLvlTitle-sc-12g27mg-0').click()

# Iteratively scroll down - load all content
last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1.0)

    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# TODO: Get list of relevant competitions
comp_list = ["1. německá liga", "1. česká liga", "1. španělská liga", "1. anglická liga", "1. francouzská liga", "1. italská liga"]

# Loop over competitions
for comp_name in comp_list:
    hide_tipsport_consent_banner(driver)

    try:
        Wait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(text(), '" + comp_name + "')]")))
        driver.find_element(By.XPATH, "//div[contains(text(), '" + comp_name + "')]").click()
    except TimeoutException:
        print(f"Error: Element containing competition text '{comp_name}' not found within 5 seconds.")
        continue

    # Loop until all matches of the competition found
    matches_texts_set = set()
    matches_list = []
    while True:
        Wait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'o-matchRow')))
        matches = driver.find_elements(By.CLASS_NAME, 'o-matchRow')
        if not matches:
            break

        for match_row in matches:
            try:
                tmp = match_row.text.split("\n")[0]
                print(f"...trying match {tmp}")

                # Get only true matches (filter out season bets etc.)
                true_matches = match_row.find_elements(By.CSS_SELECTOR, ".m-matchRowOdds--countOpp5")

                if len(true_matches) > 0:
                    btn_rates = match_row.find_elements(By.CSS_SELECTOR, ".btnRate")

                    if len(btn_rates) > 0:
                        odds_values = []

                        for btn in btn_rates:
                            odd_value = float(btn.text)
                            odds_values.append(odd_value)

                        if match_row.text not in matches_texts_set:
                            print("Unique match with 5 odds buttons found.")
                            matches_texts_set.add(match_row.text)

                            # TODO: Filter for odds
                            prob_draw_no_marg = (1 / odds_values[2]) - (1 / odds_values[2]) * (
                                    ((1 / odds_values[0] + 1 / odds_values[2] + 1 / odds_values[4]) - 1) / (
                                    1 / odds_values[0] + 1 / odds_values[2] + 1 / odds_values[4]))
                            if prob_draw_no_marg < 0.28:
                                print(f"Value {prob_draw_no_marg} does not satisfy odds conditions.")
                                continue  # does not satisfy odds condition
                            else:
                                matches_list.append(match_row)

                                # TODO: Filter for betting opportunities
                                # Open betting opportunities
                                match_row.click()
                                try:
                                    print(f"Testing if Počet gólů v zápase in match {match_row.text}...")
                                    # "Počet gólů v zápase"
                                    Wait(driver, 2).until(EC.presence_of_element_located((By.XPATH,
                                                                                            "//div[starts-with(@class, 'eventTable') and .//div[contains(text(), 'Počet gólů v zápasu')]]")))
                                    opportunity_elem = driver.find_element(By.XPATH,
                                                                           "//div[starts-with(@class, 'eventTable') and .//div[contains(text(), 'Počet gólů v zápasu')]]")

                                    # Méně než 3.5"
                                    try:
                                        element = driver.find_element(By.XPATH,
                                                                      "//div[contains(@class, 'tdEventTable opportunity') and .//span[contains(@class, 'name')]/div[contains(text(), 'Méně než 3.5')]]")
                                        value_span = element.find_element(By.XPATH, ".//span[contains(@class, 'value')]")

                                        # Get the wanted odd value
                                        value = float(value_span.text)
                                        print(f"Match to bet: {match_row.text}, Value to bet: {value}")
                                    except NoSuchElementException:
                                        pass
                                except TimeoutException:
                                    pass
                    else:
                        print("...no odds found")
                else:
                    print("...no true matches found")
            except StaleElementReferenceException:
                continue  # handle potential staleness of the match_row reference
        break

    print(f"Found {len(matches_list)} matches for the competition {comp_name}")

driver.quit()
