import time

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as Wait

from .utils import dismiss_cookie_banner, hide_sdk_banner


def show_all_matches(driver, max_clicks=200, settle_wait=1.0):
    """
    Click "Show more matches" (or "Show more") until no more matches appear.
    Uses the visible match tiles count to decide when to stop.
    """

    def match_count():
        # both older and newer layouts
        elems = driver.find_elements(
            By.CSS_SELECTOR, ".soccer .event__match--static, a.event__match, div.event__match--static"
        )
        return len(elems)

    prev = -1
    same_count_hits = 0
    for _ in range(max_clicks):
        # try to find any button variant
        btns = driver.find_elements(
            By.XPATH,
            "//a[@data-testid='wcl-buttonLink'][.//span[contains(translate(normalize-space(),"
            "'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'show more')]]",
        )
        if not btns:
            # some comps use a different markup; try a generic fallback
            btns = driver.find_elements(By.XPATH, "//a[.//span[contains(., 'Show more')]]")

        # if no button visible, we may already be at the end
        if not btns:
            break

        btn = btns[0]
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
        time.sleep(0.25)
        driver.execute_script("arguments[0].click();", btn)

        # allow new chunk to render + trigger lazy load
        time.sleep(settle_wait)
        driver.execute_script("window.scrollBy(0, 400);")
        time.sleep(0.25)

        cur = match_count()
        if cur == prev:
            same_count_hits += 1
            if same_count_hits >= 2:
                break
        else:
            same_count_hits = 0
        prev = cur

    print("All matches found successfully. Continuing...")


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
                (By.CSS_SELECTOR, 'a[href="/football/' + self.country2 + "/" + self.name2 + '/"].lmc__templateHref')
            )
        )
        driver.find_element(
            By.CSS_SELECTOR, 'a[href="/football/' + self.country2 + "/" + self.name2 + '/"].lmc__templateHref'
        ).click()

        # Archive -> FORTUNA:LIGA 2022/2023 -> Results
        hide_sdk_banner(driver)
        Wait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//div[@class="heading__name" and text()="' + self.name1 + '"]'))
        )
        driver.find_element(
            By.CSS_SELECTOR,
            'a[href="/football/' + self.country2 + "/" + self.name2 + '/archive/"]#li5.tabs__tab.archive',
        ).click()

        hide_sdk_banner(driver)
        if self.finished is True:
            Wait(driver, 10).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        'a.archive__text.archive__text--clickable[href="/football/'
                        + self.country2
                        + "/"
                        + self.name2
                        + "-"
                        + self.season
                        + '/"]',
                    )
                )
            )
            driver.find_element(
                By.CSS_SELECTOR,
                'a.archive__text.archive__text--clickable[href="/football/'
                + self.country2
                + "/"
                + self.name2
                + "-"
                + self.season
                + '/"]',
            ).click()

            hide_sdk_banner(driver)
            Wait(driver, 10).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        'a[href="/football/'
                        + self.country2
                        + "/"
                        + self.name2
                        + "-"
                        + self.season
                        + '/results/"]#li2.tabs__tab.results',
                    )
                )
            )
            driver.find_element(
                By.CSS_SELECTOR,
                'a[href="/football/'
                + self.country2
                + "/"
                + self.name2
                + "-"
                + self.season
                + '/results/"]#li2.tabs__tab.results',
            ).click()
        else:
            Wait(driver, 10).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        'a.archive__text.archive__text--clickable[href="/football/'
                        + self.country2
                        + "/"
                        + self.name2
                        + '/"]',
                    )
                )
            )
            driver.find_element(
                By.CSS_SELECTOR,
                'a.archive__text.archive__text--clickable[href="/football/' + self.country2 + "/" + self.name2 + '/"]',
            ).click()

            hide_sdk_banner(driver)
            Wait(driver, 10).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        'a[href="/football/' + self.country2 + "/" + self.name2 + '/results/"]#li2.tabs__tab.results',
                    )
                )
            )
            driver.find_element(
                By.CSS_SELECTOR,
                'a[href="/football/' + self.country2 + "/" + self.name2 + '/results/"]#li2.tabs__tab.results',
            ).click()

        hide_sdk_banner(driver)
        dismiss_cookie_banner(driver)
        show_all_matches(driver)
        print("All matches found successfully. Continuing...")
