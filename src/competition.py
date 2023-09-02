import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as Wait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from utils import hide_advert_banner, hide_sdk_banner


class CompSeason:
    def __init__(self):
        self.country1 = None
        self.country2 = None
        self.name1 = None
        self.name2 = None
        self.season = None
        self.finished = None
        self.num_of_matches_expected = None

    def load_comp_season_match_page(self, driver):
        # Show more countries -> Czech Republic -> FORTUNA:LIGA (for instance)
        time.sleep(1.5)
        Wait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "lmc__itemMore")))
        driver.find_element(By.CLASS_NAME, "lmc__itemMore").click()

        Wait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//span[text()='" + self.country1 + "']")))
        driver.find_element(By.XPATH, "//span[text()='" + self.country1 + "']").click()

        Wait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'a[href="/football/' + self.country2 + '/' + self.name2 + '/"].lmc__templateHref')))
        driver.find_element(By.CSS_SELECTOR,
                            'a[href="/football/' + self.country2 + '/' + self.name2 + '/"].lmc__templateHref').click()

        # Archive -> FORTUNA:LIGA 2022/2023 -> Results
        hide_sdk_banner(driver)
        Wait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//div[@class="heading__name" and text()="' + self.name1 + '"]')))
        driver.find_element(By.CSS_SELECTOR,
                            'a[href="/football/' + self.country2 + '/' + self.name2 + '/archive/"]#li5.tabs__tab.archive').click()

        hide_sdk_banner(driver)
        if self.finished is True:
            Wait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR,
                                                               'a.archive__text.archive__text--clickable[href="/football/' + self.country2 + '/' + self.name2 + '-' + self.season + '/"]')))
            driver.find_element(By.CSS_SELECTOR,
                            'a.archive__text.archive__text--clickable[href="/football/' + self.country2 + '/' + self.name2 + '-' + self.season + '/"]').click()

            hide_sdk_banner(driver)
            Wait(driver, 10).until(EC.presence_of_element_located(
                (By.CSS_SELECTOR,
                 'a[href="/football/' + self.country2 + '/' + self.name2 + '-' + self.season + '/results/"]#li2.tabs__tab.results')))
            driver.find_element(By.CSS_SELECTOR,
                                'a[href="/football/' + self.country2 + '/' + self.name2 + '-' + self.season + '/results/"]#li2.tabs__tab.results').click()
        else:
            Wait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR,
                                                                   'a.archive__text.archive__text--clickable[href="/football/' + self.country2 + '/' + self.name2 + '/"]')))
            driver.find_element(By.CSS_SELECTOR,
                                'a.archive__text.archive__text--clickable[href="/football/' + self.country2 + '/' + self.name2 + '/"]').click()

            hide_sdk_banner(driver)
            Wait(driver, 10).until(EC.presence_of_element_located(
                (By.CSS_SELECTOR,
                 'a[href="/football/' + self.country2 + '/' + self.name2 + '/results/"]#li2.tabs__tab.results')))
            driver.find_element(By.CSS_SELECTOR,
                                'a[href="/football/' + self.country2 + '/' + self.name2 + '/results/"]#li2.tabs__tab.results').click()

        # Show more matches -> Show more matches... (max. 3 times)
        hide_sdk_banner(driver)

        try:
            Wait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a.event__more.event__more--static[href="#"]')))
            driver.find_element(By.CSS_SELECTOR, 'a.event__more.event__more--static[href="#"]').click()
            try:
                Wait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'a.event__more.event__more--static[href="#"]')))
                hide_advert_banner(driver)
                driver.find_element(By.CSS_SELECTOR, 'a.event__more.event__more--static[href="#"]').click()
                try:
                    Wait(driver, 10).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, 'a.event__more.event__more--static[href="#"]')))
                    hide_advert_banner(driver)
                    driver.find_element(By.CSS_SELECTOR, 'a.event__more.event__more--static[href="#"]').click()
                except (TimeoutException, NoSuchElementException):
                    print("Warning: Element not present or timeout exceeded.")

            except (TimeoutException, NoSuchElementException):
                print("Warning: Element not present or timeout exceeded.")

        except (TimeoutException, NoSuchElementException):
            print("Warning: Element not present or timeout exceeded.")

        finally:
            print("All matches found successfully. Continuing...")
