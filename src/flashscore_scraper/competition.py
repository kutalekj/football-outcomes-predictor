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
        dismiss_cookie_banner(driver)
        hide_sdk_banner(driver)
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
        def safe_click(by, sel, wait_time=10, scroll=True, js_first=True, description="element"):
            """
            Wait until the element is both present and clickable, dismiss overlays,
            scroll into view, then click (JS first, fallback to native click).
            """
            # wait until Selenium thinks it's clickable (not just present)
            elem = Wait(driver, wait_time).until(EC.element_to_be_clickable((by, sel)))

            # keep trying to clear banners right before click
            dismiss_cookie_banner(driver)
            hide_sdk_banner(driver)

            if scroll:
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", elem)
                time.sleep(0.2)

            # try JS click to avoid intercept
            if js_first:
                try:
                    driver.execute_script("arguments[0].click();", elem)
                    time.sleep(0.3)
                    return
                except Exception:
                    pass

            # fallback normal click
            try:
                elem.click()
                time.sleep(0.3)
            except Exception as e:
                # last resort: raise a clearer message so we know where it failed
                raise RuntimeError(f"Click intercepted on {description}: {e}")

        # make sure no banners before starting
        dismiss_cookie_banner(driver)
        hide_sdk_banner(driver)
        time.sleep(1.0)

        # 1) Show more countries (“All countries / Show more” row)
        btn = Wait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "lmc__itemMore")))
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
        dismiss_cookie_banner(driver)
        hide_sdk_banner(driver)
        try:
            driver.execute_script("arguments[0].click();", btn)  # JS click avoids intercept
            time.sleep(0.3)
        except Exception:
            # fallback to native click
            dismiss_cookie_banner(driver)
            hide_sdk_banner(driver)
            btn.click()
            time.sleep(0.3)

        # 2) Click the desired country (self.country1)
        safe_click(
            By.XPATH,
            f"//span[normalize-space()='{self.country1}']",
            wait_time=10,
            description=f"country '{self.country1}'",
        )

        # 3) Click the league (self.name2) under that country
        league_href = f"/football/{self.country2}/{self.name2}/"
        safe_click(
            By.CSS_SELECTOR,
            f'a[href="{league_href}"].lmc__templateHref',
            wait_time=10,
            description=f"league '{self.name2}'",
        )

        # 4) Click Archive tab
        # We first wait for the heading with competition name so we know we're on the league page.
        Wait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, f'//div[@class="heading__name" and normalize-space()="{self.name1}"]')
            )
        )
        archive_selector = f'a[href="/football/{self.country2}/{self.name2}/archive/"]#li5.tabs__tab.archive'
        safe_click(
            By.CSS_SELECTOR,
            archive_selector,
            wait_time=10,
            description="archive tab",
        )

        # 5) Choose season and go to Results
        if self.finished is True:
            # finished season is like /football/england/premier-league-2024-2025/
            season_href = f"/football/{self.country2}/{self.name2}-{self.season}/"
            safe_click(
                By.CSS_SELECTOR,
                f'a.archive__text.archive__text--clickable[href="{season_href}"]',
                wait_time=10,
                description=f"season '{self.season}'",
            )

            results_href = f"/football/{self.country2}/{self.name2}-{self.season}/results/"
            safe_click(
                By.CSS_SELECTOR,
                f'a[href="{results_href}"]#li2.tabs__tab.results',
                wait_time=10,
                description="results tab (finished)",
            )

        else:
            # current season still at /football/england/premier-league/
            current_href = f"/football/{self.country2}/{self.name2}/"
            safe_click(
                By.CSS_SELECTOR,
                f'a.archive__text.archive__text--clickable[href="{current_href}"]',
                wait_time=10,
                description="current season link",
            )

            results_current_href = f"/football/{self.country2}/{self.name2}/results/"
            safe_click(
                By.CSS_SELECTOR,
                f'a[href="{results_current_href}"]#li2.tabs__tab.results',
                wait_time=10,
                description="results tab (current)",
            )

        # 6) Expand all matches on the results page
        dismiss_cookie_banner(driver)
        hide_sdk_banner(driver)
        show_all_matches(driver)
        print("All matches found successfully. Continuing...")
