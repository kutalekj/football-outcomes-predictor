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
        self.slug = None  # e.g. "isl" for ISL, or None for Premier League

    def load_comp_season_match_page(self, driver):
        def safe_click(by, sel, wait_time=10, scroll=True, js_first=True, description="element"):
            elem = Wait(driver, wait_time).until(EC.element_to_be_clickable((by, sel)))

            dismiss_cookie_banner(driver)
            hide_sdk_banner(driver)

            if scroll:
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", elem)
                time.sleep(0.1)

            if js_first:
                try:
                    driver.execute_script("arguments[0].click();", elem)
                    time.sleep(0.15)
                    return
                except Exception:
                    pass

            try:
                elem.click()
                time.sleep(0.15)
            except Exception as e:
                raise RuntimeError(f"Click intercepted on {description}: {e}")

        # --- 0) make sure overlay banners are gone
        dismiss_cookie_banner(driver)
        hide_sdk_banner(driver)
        time.sleep(0.5)

        # --- 1) Click "Show more countries" to expand the full left list
        btn = Wait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "lmc__itemMore")))
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
        dismiss_cookie_banner(driver)
        hide_sdk_banner(driver)
        try:
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(0.3)
        except Exception:
            dismiss_cookie_banner(driver)
            hide_sdk_banner(driver)
            btn.click()
            time.sleep(0.3)

        #
        # --- 2) Scroll the left menu, find the country ("India"), expand it
        #

        # 2a. After "Show more", the menu re-renders. We cannot reuse 'btn' (stale).
        #     Instead, wait for any country block to exist, then grab its ancestor
        #     as the scroll container.
        block_el = Wait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class,'lmc__block')]"))
        )

        # The parent scrollable container of these 'lmc__block's:
        menu_container = block_el.find_element(By.XPATH, "./ancestor::div[contains(@class,'lmc')][1]")

        # 2b. XPath for the country <a> that wraps the span "India"
        country_xpath = (
            f"//a[contains(@class,'lmc__element') and "
            f".//span[@class='lmc__elementName' and normalize-space()='{self.country1}']]"
        )

        country_el = None

        # 2c. Scroll the container until we see the country element in the DOM
        for _ in range(20):
            try:
                country_el = driver.find_element(By.XPATH, country_xpath)
                break
            except Exception:
                driver.execute_script("arguments[0].scrollBy(0, 400);", menu_container)
                time.sleep(0.2)

        if country_el is None:
            raise RuntimeError(f"Could not find country '{self.country1}' in left menu after scrolling.")

        # 2d. Scroll the country element into view and JS-click it to expand its leagues
        dismiss_cookie_banner(driver)
        hide_sdk_banner(driver)
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", country_el)
        time.sleep(0.15)

        try:
            driver.execute_script("arguments[0].click();", country_el)
            time.sleep(0.3)
        except Exception:
            try:
                country_el.click()
                time.sleep(0.3)
            except Exception as e:
                raise RuntimeError(f"Could not click country '{self.country1}': {e}")

        #
        # --- 3) Now wait for the ISL link to appear under India
        #
        league_href_menu = self._leftmenu_href()  # e.g. "/football/india/isl/"
        league_sel = f"a.lmc__templateHref[href='{league_href_menu}']"

        league_el = None
        for _ in range(20):
            try:
                league_el = driver.find_element(By.CSS_SELECTOR, league_sel)
                break
            except Exception:
                # Scroll a bit more in case submenu rendered below current viewport
                driver.execute_script("arguments[0].scrollBy(0, 300);", menu_container)
                time.sleep(0.2)

        if league_el is None:
            raise RuntimeError(
                f"Could not find league '{self.name1}' with href {league_href_menu} under {self.country1}."
            )

        # 3b. Scroll league link into view and JS-click it
        dismiss_cookie_banner(driver)
        hide_sdk_banner(driver)
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", league_el)
        time.sleep(0.15)

        try:
            driver.execute_script("arguments[0].click();", league_el)
            time.sleep(0.3)
        except Exception:
            try:
                league_el.click()
                time.sleep(0.3)
            except Exception as e:
                raise RuntimeError(f"Could not click league '{self.name1}': {e}")

        # --- 4) Wait for league header on the league page
        # After clicking the league, Flashscore might redirect to /isl/
        # Header looks like:
        #   <div class="heading__name">ISL</div>
        expected_heading = self.name1

        Wait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, f'//div[@class="heading__name" and normalize-space()="{expected_heading}"]')
            )
        )

        # --- 5) Navigate to the correct Results page for the desired season
        # finished == True  → go via Archive -> pick season -> Results
        # finished == False → go straight to current Results

        if self.finished:
            # click Archive tab
            archive_href = self._archive_href()
            safe_click(
                By.CSS_SELECTOR,
                f'a[href="{archive_href}"]',
                wait_time=10,
                description="archive tab",
            )

            # click the season row in archive
            season_root_href = self._season_root_href()
            safe_click(
                By.CSS_SELECTOR,
                f'a.archive__text.archive__text--clickable[href="{season_root_href}"]',
                wait_time=10,
                description=f"season '{self.season}'",
            )

            # now click Results for that season
            season_results_href = self._season_results_href()
            safe_click(
                By.CSS_SELECTOR,
                f'a[href="{season_results_href}"]',
                wait_time=10,
                description="results tab (finished season)",
            )

        else:
            # go directly to Results of the current season
            current_results_href = self._results_href_current()
            safe_click(
                By.CSS_SELECTOR,
                f'a[href="{current_results_href}"]',
                wait_time=10,
                description="results tab (current season)",
            )

        # --- 6) Expand all matches on the Results page
        dismiss_cookie_banner(driver)
        hide_sdk_banner(driver)
        show_all_matches(driver)
        print("All matches found successfully. Continuing...")

    def _root_href(self):
        """
        Return the base href for this competition's main page.
        For ISL -> "/isl/"
        For Premier League -> f"/football/{self.country2}/{self.name2}/"
        """
        if self.slug:
            return f"/{self.slug.strip('/')}/"
        return f"/football/{self.country2}/{self.name2}/"

    def _results_href_current(self):
        """
        Where to click to get match results for the current/selected season.
        ISL uses /isl/results/
        Premier League uses /football/england/premier-league/results/
        """
        root = self._root_href()
        return root + "results/"

    def _archive_href(self):
        """
        Where the Archive tab lives.
        ISL: /isl/archive/
        Premier League: /football/england/premier-league/archive/
        """
        root = self._root_href()
        return root + "archive/"

    def _season_root_href(self):
        """
        Where a specific season lives *after selecting in Archive*.
        ISL just links the current season back to /isl/.
        Premier League links e.g. /football/england/premier-league-2024-2025/
        """
        if self.slug:
            # ISL case: archive "season" link points back to base /isl/
            return self._root_href()
        return f"/football/{self.country2}/{self.name2}-{self.season}/"

    def _season_results_href(self):
        """
        Where the 'Results' tab is for that season.
        ISL: /isl/results/
        Premier League: /football/england/premier-league-2024-2025/results/
        """
        # derive from season root
        base = self._season_root_href()
        return base + "results/"

    def _leftmenu_href(self):
        """
        The href used in the LEFT MENU for this competition.
        We know from your HTML that under India it looks like:
        <a href="/football/india/isl/" class="lmc__templateHref">ISL</a>

        For Premier League under England it's:
        <a href="/football/england/premier-league/" class="lmc__templateHref">Premier League</a>

        So this ALWAYS uses the country/name form, never the slug.
        """
        return f"/football/{self.country2}/{self.name2}/"
