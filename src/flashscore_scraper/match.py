import math
import re
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse, urlunparse

import numpy as np
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as Wait


def _split_url_parts(url: str):
    """
    Returns (base_no_query, mid_or_None).
    base_no_query = https://.../match/football/<home>/<away>   (no trailing slash)
    """
    p = urlparse(url)
    path = p.path  # no query
    # cut at first of /summary or /odds
    path = re.sub(r"/(?:summary|odds)(?:/.*)?$", "", path)
    base = urlunparse((p.scheme, p.netloc, path.rstrip("/"), "", "", ""))
    qs = parse_qs(p.query or "")
    mid = None
    if "mid" in qs and qs["mid"]:
        mid = qs["mid"][0]
    return base, mid


def _make_url(base: str, tail: str, mid: str | None):
    """Join base + tail and add ?mid=... if available."""
    if not tail.startswith("/"):
        tail = "/" + tail
    url = base.rstrip("/") + tail
    if mid:
        url += "?mid=" + mid
    return url


def _split_title_start_end(a_el):
    """
    a_el.get_attribute('title') is like '4.86 » 5.90' or sometimes empty.
    Returns (start, end) as floats or (None, None).
    """
    t = (a_el.get_attribute("title") or "").strip()
    if "»" in t:
        parts = [p.strip().replace(",", ".") for p in t.split("»")]
        try:
            return float(parts[0]), float(parts[-1])
        except Exception:
            return (None, None)
    # fallback: no title → take visible text as 'end', unknown start
    try:
        end = float((a_el.text or a_el.get_attribute("textContent") or "").strip().replace(",", "."))
    except Exception:
        end = None
    return (None, end)


def _ensure_odds_table_loaded(driver, timeout=5):
    # Wait for odds rows to exist on the CURRENT page (don't navigate again)
    Wait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".ui-table__row .oddsCell__odd")))


def open_odds_tab(driver):
    # Fast path: go straight to the 1X2 odds URL using base + mid
    base, mid = _split_url_parts(driver.current_url)
    direct_url = _make_url(base, "odds/1x2-odds/", mid)
    try:
        if not driver.current_url.startswith(direct_url):
            driver.get(direct_url)
        # ensure odds table or widget is there quickly
        Wait(driver, 4).until(
            EC.any_of(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".ui-table__row .oddsCell__odd")),
                EC.presence_of_element_located((By.CSS_SELECTOR, ".wclOddsContent .wclOddsRow")),
            )
        )
        return
    except TimeoutException:
        pass  # fallback below

    # Fallback: try clicking the 'Odds' tab in case they're testing a different structure
    try:
        btn = Wait(driver, 3).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[@role='tab' and translate(normalize-space(.),'odS','ods')='odds']")
            )
        )
        if btn.get_attribute("data-selected") != "true":
            driver.execute_script("arguments[0].click();", btn)
        # short wait for odds content
        Wait(driver, 3).until(
            EC.any_of(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".ui-table__row .oddsCell__odd")),
                EC.presence_of_element_located((By.CSS_SELECTOR, ".wclOddsContent .wclOddsRow")),
            )
        )
    except Exception:
        # If all else fails we'll still continue; scrape_* will just miss odds for this match
        pass


def _open_stats_and_get_rows(driver, timeout=6):
    # Best: navigate to a well-formed stats URL using base + mid
    base, mid = _split_url_parts(driver.current_url)
    stats_url = _make_url(base, "summary/stats", mid)
    if not driver.current_url.startswith(stats_url):
        driver.get(stats_url)

    # wait for at least one stat row
    rows = Wait(driver, timeout).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='wcl-statistics']"))
    )
    return rows


def _parse_number(text):
    t = text.strip()
    # strip percent
    if t.endswith("%"):
        try:
            return float(t[:-1])
        except ValueError:
            return None
    # integer / float
    try:
        return int(t)
    except ValueError:
        try:
            return float(t)
        except ValueError:
            return None


def scrape_match_result_odds(driver, target):
    """
    Fill 1X2 start/end odds for Tipsport (49) and Fortuna (46).
    Returns a dict like {"tipsport": True/False, "fortuna": True/False}
    indicating whether we managed to fill that bookmaker from this pass.
    """
    open_odds_tab(driver)

    filled = {"tipsport": False, "fortuna": False}

    def _set(prefix, s1, e1, s0, e0, s2, e2):
        setattr(target, f"odd_{prefix}_1_start", s1 or "")
        setattr(target, f"odd_{prefix}_1_end", e1 or "")
        setattr(target, f"odd_{prefix}_0_start", s0 or "")
        setattr(target, f"odd_{prefix}_0_end", e0 or "")
        setattr(target, f"odd_{prefix}_2_start", s2 or "")
        setattr(target, f"odd_{prefix}_2_end", e2 or "")

    def _handle_new_widget(bm_id, prefix):
        try:
            block = driver.find_element(
                By.XPATH,
                f"//div[contains(@class,'wclOddsContent')]/div[contains(@class,'odds')]"
                f"[.//a[contains(@href,'/bookmaker/{bm_id}/')]]",
            )
            row = block.find_element(By.CSS_SELECTOR, ".wclOddsRow")
            cells = row.find_elements(By.CSS_SELECTOR, "[data-testid='wcl-oddsCell']")
            if len(cells) >= 3:
                s1, e1 = _split_title_start_end(cells[0])  # home
                s0, e0 = _split_title_start_end(cells[1])  # draw
                s2, e2 = _split_title_start_end(cells[2])  # away
                _set(prefix, s1, e1, s0, e0, s2, e2)
                return True
        except Exception:
            pass
        return False

    def _handle_old_table(bm_id, prefix):
        try:
            row = driver.find_element(
                By.XPATH, f"//div[contains(@class,'ui-table__row')][.//div[@data-analytics-bookmaker-id='{bm_id}']]"
            )
            a_cells = row.find_elements(By.CSS_SELECTOR, "a.oddsCell__odd")
            if len(a_cells) >= 3:
                s1, e1 = _split_title_start_end(a_cells[0])
                s0, e0 = _split_title_start_end(a_cells[1])
                s2, e2 = _split_title_start_end(a_cells[2])
                _set(prefix, s1, e1, s0, e0, s2, e2)
                return True
        except Exception:
            pass
        return False

    for bm_id, prefix in (("49", "tipsport"), ("46", "fortuna")):
        ok = _handle_new_widget(bm_id, prefix)
        if not ok:
            ok = _handle_old_table(bm_id, prefix)
        filled[prefix] = ok

    return filled


def open_over_under_tab(driver):
    # Prefer direct navigation to avoid tab DOM variations
    base, mid = _split_url_parts(driver.current_url)
    driver.get(_make_url(base, "odds/over-under/full-time/", mid))
    # Wait for either old table or new widget to appear
    try:
        Wait(driver, 4).until(
            EC.any_of(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".ui-table__row .oddsCell__odd")),
                EC.presence_of_element_located((By.CSS_SELECTOR, ".wclOddsContent .wclOddsRow")),
            )
        )
    except TimeoutException:
        # Retry once via clicking the subtab if it exists
        try:
            ou = Wait(driver, 3).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[@data-testid='wcl-tab' and normalize-space()='Over/Under']")
                )
            )
            driver.execute_script("arguments[0].click();", ou)

            # wait for actual odds rows instead of sleeping
            Wait(driver, 4).until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".ui-table__row .oddsCell__odd")),
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".wclOddsContent .wclOddsRow")),
                )
            )
        except Exception:
            pass


def scrape_over_under_totals(driver, target, totals=(1.5, 2.5, 3.5)):
    """
    For each total T in totals and for each bookmaker (49 Tipsport, 46 Fortuna), fills:
      ou{T*10}_tipsport_over_start/end, ou{T*10}_tipsport_under_start/end
      ou{T*10}_fortuna_over_start/end,  ou{T*10}_fortuna_under_start/end
    """

    def _handle_ou_new_widget(bookmaker_id, prefix, T):
        # Find a block for the bookmaker and rows with odds cells
        try:
            block = driver.find_element(
                By.XPATH,
                f"//div[contains(@class,'wclOddsContent')]"
                f"//div[contains(@class,'odds')][.//a[contains(@href,'/bookmaker/{bookmaker_id}/')]]",
            )
        except Exception:
            return False

        # Find a row with the threshold label T (1.5 or 1,5)
        labels = [f"{T:.1f}", f"{T:.1f}".replace(".", ",")]
        rows = block.find_elements(By.CSS_SELECTOR, ".wclOddsRow")
        for r in rows:
            try:
                has_T = any(
                    (e.text or "").strip() in labels
                    for e in r.find_elements(By.CSS_SELECTOR, "[data-testid='wcl-oddsValue']")
                )
                if not has_T:
                    continue
                cells = r.find_elements(By.CSS_SELECTOR, "[data-testid='wcl-oddsCell']")
                if len(cells) < 2:
                    continue
                so, eo = _split_title_start_end(cells[0])  # Over
                su, eu = _split_title_start_end(cells[1])  # Under
                key = int(T * 10)
                setattr(target, f"ou{key}_{prefix}_over_start", so or "")
                setattr(target, f"ou{key}_{prefix}_over_end", eo or "")
                setattr(target, f"ou{key}_{prefix}_under_start", su or "")
                setattr(target, f"ou{key}_{prefix}_under_end", eu or "")
                return True
            except Exception:
                continue
        return False

    open_over_under_tab(driver)

    def _label_variants(T):
        # both dot and comma, e.g. "1.5" or "1,5"
        s = f"{T:.1f}"
        return [s.replace(".", ","), s]

    def _handle_ou_for(bookmaker_id, prefix, T):
        rows = driver.find_elements(
            By.XPATH, f"//div[contains(@class,'ui-table__row')][.//div[@data-analytics-bookmaker-id='{bookmaker_id}']]"
        )
        for row in rows:
            try:
                # try both 1.5 and 1,5
                ok = False
                for lab in _label_variants(T):
                    els = row.find_elements(
                        By.XPATH, f".//span[@data-testid='wcl-oddsValue' and normalize-space()='{lab}']"
                    )
                    if els:
                        ok = True
                        break
                if not ok:
                    continue

                a_cells = row.find_elements(By.CSS_SELECTOR, "a.oddsCell__odd")
                if len(a_cells) < 2:
                    continue

                so, eo = _split_title_start_end(a_cells[0])  # Over
                su, eu = _split_title_start_end(a_cells[1])  # Under
                key = int(T * 10)  # 1.5 -> 15
                setattr(target, f"ou{key}_{prefix}_over_start", so or "")
                setattr(target, f"ou{key}_{prefix}_over_end", eo or "")
                setattr(target, f"ou{key}_{prefix}_under_start", su or "")
                setattr(target, f"ou{key}_{prefix}_under_end", eu or "")
                break
            except Exception:
                continue

    for T in totals:
        if not _handle_ou_new_widget("49", "tipsport", T):
            _handle_ou_for("49", "tipsport", T)
        if not _handle_ou_new_widget("46", "fortuna", T):
            _handle_ou_for("46", "fortuna", T)


def _parse_pct(text):
    # "69%" -> 69 ; also handles "87% (548/633)" (we only need percentage)
    return int(text.strip().split("%")[0])


def scrape_stats(driver, target):
    label_to_attr = {
        "Expected Goals (xG)": ("expected_goals_home", "expected_goals_away"),
        "Ball Possession": ("possession_home", "possession_away"),
        "Total shots": ("shots_total_home", "shots_total_away"),
        "Shots on target": ("shots_on_goal_home", "shots_on_goal_away"),
        "Corner Kicks": ("corner_kicks_home", "corner_kicks_away"),
        "Passes": ("pass_success_home", "pass_success_away"),
    }

    try:
        rows = _open_stats_and_get_rows(driver)
    except Exception as e:
        print(f"⚠️ Statistics scraping failed: {type(e).__name__}: {repr(e)}")
        return

    def _num(t: str):
        t = t.strip()
        if not t:
            return None
        # drop "(x/y)" trailing info if present
        t = re.sub(r"\s*\([^)]*\)\s*$", "", t)
        if t.endswith("%"):
            t = t[:-1]
        t = t.replace(",", ".")
        try:
            if "." in t:
                return float(t)
            return int(t)
        except ValueError:
            return None

    for row in rows:
        try:
            cat = row.find_element(By.CSS_SELECTOR, "[data-testid='wcl-statistics-category'] > strong").text.strip()
            vals = row.find_elements(
                By.CSS_SELECTOR, "[data-testid='wcl-statistics-value'] strong[data-testid='wcl-scores-simple-text-01']"
            )
            if len(vals) < 2:
                continue
            home_text, away_text = vals[0].text.strip(), vals[1].text.strip()

            if cat in label_to_attr:
                hk, ak = label_to_attr[cat]
                setattr(target, hk, _num(home_text))
                setattr(target, ak, _num(away_text))
        except Exception:
            continue


def _open_odds_tab_1x2(driver, timeout=5):
    base, mid = _split_url_parts(driver.current_url)
    driver.get(_make_url(base, "odds/1x2-odds/", mid))
    Wait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".ui-table__row .oddsCell__odd")))


def _parse_start_end_from_title(a_el):
    # title="1.36 » 1.34" OR sometimes missing
    title = a_el.get_attribute("title") or ""
    m = re.search(r"([\d.]+)\s*»\s*([\d.]+)", title)
    end_val_text = a_el.find_element(By.CSS_SELECTOR, "span").text.strip()
    end_val = float(end_val_text) if end_val_text else None
    if m:
        start_val = float(m.group(1))
        end_val_from_title = float(m.group(2))
        # prefer title's end if present; otherwise use span
        end_val = end_val_from_title
    else:
        start_val = None  # will be backfilled with end later
    return start_val, end_val


def scrape_1x2_from_odds_table(driver, target, bookmaker_alt="Tipsport.cz"):
    """
    Fallback parser for old table layout.
    Assumes we're ALREADY on the 1X2 odds page.
    """
    _ensure_odds_table_loaded(driver)
    rows = driver.find_elements(By.CSS_SELECTOR, ".ui-table.oddsCell__odds .ui-table__row")
    for row in rows:
        try:
            img = row.find_element(By.CSS_SELECTOR, ".oddsCell__bookmaker img.prematchLogo")
            if (img.get_attribute("alt") or "").strip() != bookmaker_alt:
                continue
            cells = row.find_elements(By.CSS_SELECTOR, "a.oddsCell__odd")
            if len(cells) < 3:
                continue

            s1, e1 = _parse_start_end_from_title(cells[0])
            sX, eX = _parse_start_end_from_title(cells[1])
            s2, e2 = _parse_start_end_from_title(cells[2])

            target.odd_tipsport_1_start = s1
            target.odd_tipsport_0_start = sX
            target.odd_tipsport_2_start = s2
            target.odd_tipsport_1_end = e1
            target.odd_tipsport_0_end = eX
            target.odd_tipsport_2_end = e2

            # backfill
            for a, b in (
                ("odd_tipsport_1_start", "odd_tipsport_1_end"),
                ("odd_tipsport_0_start", "odd_tipsport_0_end"),
                ("odd_tipsport_2_start", "odd_tipsport_2_end"),
            ):
                if getattr(target, a) in (None, "") and getattr(target, b) not in (None, ""):
                    setattr(target, a, getattr(target, b))
            return True
        except Exception:
            continue
    return False


def _is_missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and x.strip() == "")


def _backfill_starts(row_dict: dict) -> dict:
    """
    If key ends with '_end' and corresponding '_start' is missing/empty/NaN,
    copy the end value into the start key.
    """
    for end_key, val in list(row_dict.items()):
        m = re.match(r"^(.*)_end$", end_key)
        if not m:
            continue
        start_key = m.group(1) + "_start"  # always WITH the underscore
        if _is_missing(row_dict.get(start_key)) and not _is_missing(val):
            row_dict[start_key] = val
    return row_dict


class Match:
    def __init__(self):
        self.id = None
        self.match_valid = True
        self.date_time = None

        self.team_home = None
        self.team_away = None
        self.goals_home = None
        self.goals_away = None
        self.result = None

        self.country = None
        self.competition = None
        self.season = None
        self.round = None

        self.neutral_field = None
        self.finished = None

        self.odd_tipsport_1_start = None
        self.odd_tipsport_1_end = None
        self.odd_tipsport_0_start = None
        self.odd_tipsport_0_end = None
        self.odd_tipsport_2_start = None
        self.odd_tipsport_2_end = None

        self.odd_fortuna_1_start = None
        self.odd_fortuna_1_end = None
        self.odd_fortuna_0_start = None
        self.odd_fortuna_0_end = None
        self.odd_fortuna_2_start = None
        self.odd_fortuna_2_end = None

        self.ou15_tipsport_over_start = None
        self.ou15_tipsport_over_end = None
        self.ou15_tipsport_under_start = None
        self.ou15_tipsport_under_end = None
        self.ou25_tipsport_over_start = None
        self.ou25_tipsport_over_end = None
        self.ou25_tipsport_under_start = None
        self.ou25_tipsport_under_end = None
        self.ou35_tipsport_over_start = None
        self.ou35_tipsport_over_end = None
        self.ou35_tipsport_under_start = None
        self.ou35_tipsport_under_end = None

        self.ou15_fortuna_over_start = None
        self.ou15_fortuna_over_end = None
        self.ou15_fortuna_under_start = None
        self.ou15_fortuna_under_end = None
        self.ou25_fortuna_over_start = None
        self.ou25_fortuna_over_end = None
        self.ou25_fortuna_under_start = None
        self.ou25_fortuna_under_end = None
        self.ou35_fortuna_over_start = None
        self.ou35_fortuna_over_end = None
        self.ou35_fortuna_under_start = None
        self.ou35_fortuna_under_end = None

        self.possession_home = -1
        self.possession_away = -1

        self.shots_total_home = -1
        self.shots_total_away = -1
        self.shots_on_goal_home = -1
        self.shots_on_goal_away = -1
        self.shots_off_goal_home = -1
        self.shots_off_goal_away = -1
        self.shots_blocked_home = -1
        self.shots_blocked_away = -1

        self.corner_kicks_home = -1
        self.corner_kicks_away = -1
        self.offsides_home = -1
        self.offsides_away = -1
        self.throw_ins_home = -1
        self.throw_ins_away = -1

        self.expected_goals_home = -1.0
        self.expected_goals_away = -1.0

        self.pass_success_home = -1
        self.pass_success_away = -1

    def to_dict(self):
        def emit(v):
            # blank only if None or sentinel -1/NaN
            if v is None:
                return ""
            if isinstance(v, (int, float)):
                if v == -1 or (isinstance(v, float) and (v != v)):  # NaN
                    return ""
            return v

        out = {
            "id": self.id,
            "datetime": self.date_time,
            "team_home": self.team_home,
            "team_away": self.team_away,
            "goals_home": self.goals_home,
            "goals_away": self.goals_away,
            "result": self.result,
            "country": self.country,
            "competition": self.competition,
            "season": self.season,
            "round": self.round if self.round is not None else "",
            "neutral_field": int(bool(self.neutral_field)),
            "finished": int(bool(self.finished)),
            # 1X2
            "odd_tipsport_1_start": emit(self.odd_tipsport_1_start),
            "odd_tipsport_1_end": emit(self.odd_tipsport_1_end),
            "odd_tipsport_0_start": emit(self.odd_tipsport_0_start),
            "odd_tipsport_0_end": emit(self.odd_tipsport_0_end),
            "odd_tipsport_2_start": emit(self.odd_tipsport_2_start),
            "odd_tipsport_2_end": emit(self.odd_tipsport_2_end),
            "odd_fortuna_1_start": emit(self.odd_fortuna_1_start),
            "odd_fortuna_1_end": emit(self.odd_fortuna_1_end),
            "odd_fortuna_0_start": emit(self.odd_fortuna_0_start),
            "odd_fortuna_0_end": emit(self.odd_fortuna_0_end),
            "odd_fortuna_2_start": emit(self.odd_fortuna_2_start),
            "odd_fortuna_2_end": emit(self.odd_fortuna_2_end),
            # Over/Under (1.5, 2.5, 3.5)
            "ou15_tipsport_over_start": emit(self.ou15_tipsport_over_start),
            "ou15_tipsport_over_end": emit(self.ou15_tipsport_over_end),
            "ou15_tipsport_under_start": emit(self.ou15_tipsport_under_start),
            "ou15_tipsport_under_end": emit(self.ou15_tipsport_under_end),
            "ou25_tipsport_over_start": emit(self.ou25_tipsport_over_start),
            "ou25_tipsport_over_end": emit(self.ou25_tipsport_over_end),
            "ou25_tipsport_under_start": emit(self.ou25_tipsport_under_start),
            "ou25_tipsport_under_end": emit(self.ou25_tipsport_under_end),
            "ou35_tipsport_over_start": emit(self.ou35_tipsport_over_start),
            "ou35_tipsport_over_end": emit(self.ou35_tipsport_over_end),
            "ou35_tipsport_under_start": emit(self.ou35_tipsport_under_start),
            "ou35_tipsport_under_end": emit(self.ou35_tipsport_under_end),
            "ou15_fortuna_over_start": emit(self.ou15_fortuna_over_start),
            "ou15_fortuna_over_end": emit(self.ou15_fortuna_over_end),
            "ou15_fortuna_under_start": emit(self.ou15_fortuna_under_start),
            "ou15_fortuna_under_end": emit(self.ou15_fortuna_under_end),
            "ou25_fortuna_over_start": emit(self.ou25_fortuna_over_start),
            "ou25_fortuna_over_end": emit(self.ou25_fortuna_over_end),
            "ou25_fortuna_under_start": emit(self.ou25_fortuna_under_start),
            "ou25_fortuna_under_end": emit(self.ou25_fortuna_under_end),
            "ou35_fortuna_over_start": emit(self.ou35_fortuna_over_start),
            "ou35_fortuna_over_end": emit(self.ou35_fortuna_over_end),
            "ou35_fortuna_under_start": emit(self.ou35_fortuna_under_start),
            "ou35_fortuna_under_end": emit(self.ou35_fortuna_under_end),
            # Stats
            "possession_home": emit(self.possession_home),
            "possession_away": emit(self.possession_away),
            "shots_total_home": emit(self.shots_total_home),
            "shots_total_away": emit(self.shots_total_away),
            "shots_on_goal_home": emit(self.shots_on_goal_home),
            "shots_on_goal_away": emit(self.shots_on_goal_away),
            "shots_off_goal_home": emit(self.shots_off_goal_home),
            "shots_off_goal_away": emit(self.shots_off_goal_away),
            "shots_blocked_home": emit(self.shots_blocked_home),
            "shots_blocked_away": emit(self.shots_blocked_away),
            "corners_home": emit(self.corner_kicks_home),
            "corners_away": emit(self.corner_kicks_away),
            "offsides_home": emit(self.offsides_home),
            "offsides_away": emit(self.offsides_away),
            "throw_ins_home": emit(self.throw_ins_home),
            "throw_ins_away": emit(self.throw_ins_away),
            "xg_home": emit(self.expected_goals_home),
            "xg_away": emit(self.expected_goals_away),
            "passes_accuracy_home": emit(self.pass_success_home),
            "passes_accuracy_away": emit(self.pass_success_away),
        }

        _backfill_starts(out)
        return out

    def get_match_statistics(self, driver, country, comp_name, season):
        def _find_first(driver, css_selectors, timeout=3):
            for css in css_selectors:
                try:
                    if timeout and timeout > 0:
                        return Wait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, css)))
                    else:
                        return driver.find_element(By.CSS_SELECTOR, css)
                except Exception:
                    continue
            return None

        # --- 1) Date & time ---
        dt_el = _find_first(
            driver,
            [
                ".duelParticipant__startTime > div",
                ".duelParticipant__startTime",
                "[data-testid='wcl-moment']",
                ".wcl-moment",
            ],
            timeout=4,
        )
        if not dt_el:
            raise NoSuchElementException("Could not locate match date/time element")
        raw_dt = dt_el.text.strip()
        parsed = None
        for fmt in ("%d.%m.%Y %H:%M", "%d/%m/%Y %H:%M", "%d.%m.%y %H:%M"):
            try:
                parsed = datetime.strptime(raw_dt, fmt)
                break
            except ValueError:
                continue
        self.date_time = parsed.strftime("%Y-%m-%d %H:%M") if parsed else None

        # --- 2/3) Teams ---
        self.team_home = _find_first(
            driver,
            [
                ".duelParticipant__home .participant__participantName.participant__overflow > a",
                ".duelParticipant__home .participant__participantName > a",
                ".duelParticipant__home .wcl-simpleText_2t3pL",
            ],
        ).text.strip()
        self.team_away = _find_first(
            driver,
            [
                ".duelParticipant__away .participant__participantName.participant__overflow > a",
                ".duelParticipant__away .participant__participantName > a",
                ".duelParticipant__away .wcl-simpleText_2t3pL",
            ],
        ).text.strip()
        print(f"{self.team_home} - {self.team_away}")

        # --- 0) ID ---
        self.id = f"{raw_dt}_{self.team_home}_{self.team_away}"

        # --- 4/5/6) Competition, Season, Round, Country ---
        self.country = country
        self.competition = comp_name
        self.season = season

        try:
            Wait(driver, 5).until(
                EC.presence_of_all_elements_located(
                    (
                        By.CSS_SELECTOR,
                        "li[data-testid='wcl-breadcrumbsItem'] span[data-testid='wcl-scores-overline-03']",
                    )
                )
            )
            spans = driver.find_elements(
                By.CSS_SELECTOR, "li[data-testid='wcl-breadcrumbsItem'] span[data-testid='wcl-scores-overline-03']"
            )
            txt = " | ".join(s.text for s in spans if s.text)
        except Exception:
            txt = ""
        m = re.search(r"round[^0-9]{0,5}(\d+)", txt, re.IGNORECASE)
        if not m:
            header_el = _find_first(
                driver,
                [".tournamentHeader__country > a", ".wcl-headerTournament a", ".tournamentHeader__country"],
                timeout=2,
            )
            header_text = header_el.text.strip() if header_el else ""
            m = re.search(r"[Rr]ound(?:\s*|\xa0|-)*(\d+)", header_text)
        self.round = int(m.group(1)) if m else None
        print(f"{self.competition} {self.season}: Round {self.round if self.round else 'UNKNOWN'}")

        # --- 11/12) Neutral field & finished ---
        try:
            info_box = driver.find_element(By.CSS_SELECTOR, ".infoBox__wrapper .infoBox__info").text
            self.neutral_field = "at a different stadium" in info_box
        except NoSuchElementException:
            self.neutral_field = False

        try:
            finished_elem = Wait(driver, 4).until(
                EC.presence_of_element_located((By.CLASS_NAME, "fixedHeaderDuel__detailStatus"))
            )
            finished_text = driver.execute_script("return arguments[0].innerText;", finished_elem)
            self.finished = finished_text.strip().upper() in ("FINISHED", "FT")
        except Exception:
            self.finished = False

        if not self.finished:
            print("⚠️ Unfinished match found — skipping.")
            self.match_valid = False
            return

        # --- 7/8/9) Score & result ---
        score_div = _find_first(driver, [".detailScore__wrapper"], timeout=3)
        score_spans = score_div.find_elements(By.TAG_NAME, "span")
        if len(score_spans) >= 3:
            self.goals_home = int(score_spans[0].text)
            self.goals_away = int(score_spans[2].text)
        else:
            nums = re.findall(r"\d+", score_div.text)
            if len(nums) >= 2:
                self.goals_home, self.goals_away = map(int, nums[:2])
            else:
                self.goals_home = self.goals_away = -1
        self.result = 1 if self.goals_home > self.goals_away else 0 if self.goals_home == self.goals_away else 2

        # 1X2 odds (new widget or old table)
        filled_status = scrape_match_result_odds(driver, self)

        # If Tipsport or Fortuna wasn't filled at all, try fallback parser on the SAME page
        if not (filled_status.get("tipsport") and filled_status.get("fortuna")):
            scrape_1x2_from_odds_table(driver, self)

        # Over/Under odds
        scrape_over_under_totals(driver, self, totals=(1.5, 2.5, 3.5))

        # Team statistics (this opens the Stats tab itself, robust to layout)
        scrape_stats(driver, self)

        self.match_valid = True

    @staticmethod
    def correct_zero_values(matches):
        """If a metric appears for any match, normalize missing ones to 0 (or 0.0)."""
        int_metrics = [
            "possession_home",
            "possession_away",
            "shots_total_home",
            "shots_total_away",
            "shots_on_goal_home",
            "shots_on_goal_away",
            "shots_off_goal_home",
            "shots_off_goal_away",
            "shots_blocked_home",
            "shots_blocked_away",
            "corner_kicks_home",
            "corner_kicks_away",
            "offsides_home",
            "offsides_away",
            "throw_ins_home",
            "throw_ins_away",
            "pass_success_home",
            "pass_success_away",
        ]
        float_metrics = ["expected_goals_home", "expected_goals_away"]

        for attr in int_metrics:
            if any(getattr(m, attr, -1) > -1 for m in matches):
                for m in matches:
                    if getattr(m, attr, -1) == -1:
                        setattr(m, attr, 0)

        for attr in float_metrics:
            if any((getattr(m, attr, -1.0) is not None) and (getattr(m, attr) >= 0) for m in matches):
                for m in matches:
                    if getattr(m, attr, -1.0) == -1.0:
                        setattr(m, attr, 0.0)

    @staticmethod
    def check_num_of_matches(matches, comp):
        if comp.finished is True and len(matches) != comp.num_of_matches_expected:
            raise ValueError(f"Found {len(matches)} matches, but {comp.num_of_matches_expected} was expected.")

    @staticmethod
    def drop_duplicate_matches(df):
        """Prefer rows with more non-null / non -1.0 values."""
        df_copy = df.copy()
        df_copy[df_copy == -1.0] = np.nan
        df_copy["__priority__"] = df_copy.count(axis=1)
        out = (
            df_copy.sort_values("__priority__", ascending=False)
            .drop_duplicates(keep="first")
            .drop(columns="__priority__")
        )
        # Fill back metric NaNs with -1.0 (keep string columns as-is)
        for col in out.columns:
            if out[col].dtype.kind in "biufc":
                out[col] = out[col].fillna(-1.0)
        return out


def _dismiss_overlays(driver):
    # Cookie banner
    for css in ["#onetrust-accept-btn-handler", "button#onetrust-accept-btn-handler"]:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, css)
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(0.2)
            break
        except Exception:
            pass
    # Occasionally a tooltip/banner can block clicks; ESC gets rid of some
    try:
        ActionChains(driver).send_keys(Keys.ESCAPE).perform()
    except Exception:
        pass


def _open_stats_tab(driver):
    _dismiss_overlays(driver)
    wait = Wait(driver, 12)

    # Ensure the tab bar is present before we try to click anything
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='wcl-tab']")))

    # Primary: click the real stats anchor
    try:
        a = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a[data-analytics-alias='match-statistics']")))
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", a)
        driver.execute_script("arguments[0].click();", a)
    except Exception:
        # Fallback: navigate directly
        cur = driver.current_url.rstrip("/")
        stats_url = re.sub(r"/summary(?:/.*)?$", "/summary/stats", cur)
        if not stats_url.endswith("/summary/stats"):
            stats_url += "/summary/stats"
        driver.get(stats_url)

    # Rows must appear
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='wcl-statistics']")))


def _parse_stats_rows(driver):
    # Works on the new Flashscore layout you pasted
    rows = driver.find_elements(By.CSS_SELECTOR, "[data-testid='wcl-statistics']")
    if not rows:
        raise RuntimeError("Stats rows not found")

    stats = {}
    for row in rows:
        try:
            label_el = row.find_element(By.CSS_SELECTOR, "[data-testid='wcl-statistics-category'] strong")
            label = label_el.text.strip()
            # Two values exist – first (home) and last (away)
            vals = row.find_elements(By.CSS_SELECTOR, "[data-testid='wcl-statistics-value'] strong")
            if len(vals) >= 2:
                home_raw = vals[0].text.strip()
                away_raw = vals[-1].text.strip()
            else:
                # very rare, skip row
                continue

            # Strip extra parentheses like "(344/410)"
            home = re.sub(r"\s*\([^)]*\)", "", home_raw).strip()
            away = re.sub(r"\s*\([^)]*\)", "", away_raw).strip()

            # Normalize label to a safe column name
            key = (
                label.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("%", "pct")
                .replace("/", "_")
                .replace("-", "_")
            )
            stats[f"stats_home__{key}"] = home
            stats[f"stats_away__{key}"] = away
        except Exception:
            # continue; some rows may be odd
            continue

    return stats
