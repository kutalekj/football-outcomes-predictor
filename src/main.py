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


def get_match_statistics():
    new_match = Match()

    # 1. Date & Time
    date_time = driver.find_element(By.CSS_SELECTOR, '.duelParticipant__startTime > div').text
    date_time_parsed = datetime.strptime(date_time, "%d.%m.%Y %H:%M")
    new_match.date_time = date_time_parsed.strftime("%Y-%m-%d %H:%M")

    # 2./3. Teams
    new_match.team_home = driver.find_element(By.CSS_SELECTOR,
                                              '.duelParticipant__home .participant__participantName.participant__overflow > a').text
    new_match.team_away = driver.find_element(By.CSS_SELECTOR,
                                              '.duelParticipant__away .participant__participantName.participant__overflow > a').text
    print(new_match.team_home + " - " + new_match.team_away)

    # 4./5./6. Competition, Season, Round
    new_match.competition = "FORTUNA:LIGA"
    new_match.season = "2022-2023"
    new_match.round = int(
        driver.find_element(By.CSS_SELECTOR, '.tournamentHeader__country > a').text.split('ROUND ')[1])
    print(new_match.competition + " " + new_match.season + ": Round " + str(new_match.round))

    # 7./8./9. Result, Team Goals - Home/Away
    score_div = driver.find_element(By.CSS_SELECTOR, '.detailScore__wrapper')
    score_spans = score_div.find_elements(By.TAG_NAME, 'span')

    new_match.goals_home = int(score_spans[0].text)
    new_match.goals_away = int(score_spans[2].text)

    if new_match.goals_home >= new_match.goals_away:
        if new_match.goals_home > new_match.goals_away:
            new_match.result = 0
        else:
            new_match.result = 1
    else:
        new_match.result = 2
    print(str(new_match.goals_home) + ":" + str(new_match.goals_away) + "\t(winner = " + str(new_match.result) + ")")

    # 10. Referee
    referee_div = driver.find_element(By.CSS_SELECTOR, '.section .mi__data')
    referee_div2 = referee_div.find_elements(By.TAG_NAME, 'div')[0]
    new_match.referee = referee_div2.find_element(By.CSS_SELECTOR, '.mi__item__val').text.strip()

    # 11./12. Neutral field, Finished
    try:
        match_info = driver.find_element(By.CSS_SELECTOR, '.infoBox__wrapper .infoBox__info').text
        if "at a different stadium" in match_info:
            new_match.neutral_field = True
    except NoSuchElementException:
        new_match.neutral_field = False

    finished_elem = Wait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "fixedHeaderDuel__detailStatus")))
    finished_text = driver.execute_script("return arguments[0].innerText;", finished_elem)
    new_match.finished = True if finished_text == "FINISHED" else False
    if not new_match.finished:
        raise Exception("All matches must be finished")
    print("Referee: " + new_match.referee + ", neutral field = " + str(new_match.neutral_field) + ", finished = " + str(
        new_match.finished))

    # 13.- 24. Odds (Tipsport, Fortuna)
    odds_elems = driver.find_elements(By.CSS_SELECTOR, '.oddsRowContent')
    for odd_elem, i in zip(odds_elems, range(len(odds_elems))):
        last_minute_odds = odd_elem.find_elements(By.CSS_SELECTOR, '.oddsValueInner')

        last_minute_odd1 = last_minute_odds[0].text
        last_minute_odd0 = last_minute_odds[1].text
        last_minute_odd2 = last_minute_odds[2].text

        init_odds = odd_elem.find_elements(By.CSS_SELECTOR, '.cellWrapper')

        odd1 = init_odds[0].get_attribute("title")
        init_odd1 = odd1.split(' ')[0] if odd1 != "" else last_minute_odd1
        odd0 = init_odds[1].get_attribute("title")
        init_odd0 = odd0.split(' ')[0] if odd0 != "" else last_minute_odd0
        odd2 = init_odds[2].get_attribute("title")
        init_odd2 = odd2.split(' ')[0] if odd2 != "" else last_minute_odd2

        print(str(init_odd1) + " >> " + str(last_minute_odd1) + ", " + str(init_odd0) + " >> " + str(
            last_minute_odd0) + ", " + str(init_odd2) + " >> " + str(last_minute_odd2))

        if i == 0:
            new_match.odd_tipsport_1_start = float(init_odd1)
            new_match.odd_tipsport_1_end = float(last_minute_odd1)
            new_match.odd_tipsport_0_start = float(init_odd0)
            new_match.odd_tipsport_0_end = float(last_minute_odd0)
            new_match.odd_tipsport_2_start = float(init_odd2)
            new_match.odd_tipsport_2_end = float(last_minute_odd2)
        else:
            new_match.odd_fortuna_1_start = float(init_odd1)
            new_match.odd_fortuna_1_end = float(last_minute_odd1)
            new_match.odd_fortuna_0_start = float(init_odd0)
            new_match.odd_fortuna_0_end = float(last_minute_odd0)
            new_match.odd_fortuna_2_start = float(init_odd2)
            new_match.odd_fortuna_2_end = float(last_minute_odd2)

    # Yellow/Red cards Not on pitch
    yellows_not_on_pitch_counter = 0
    reds_not_on_pitch_counter = 0

    incidents_section = driver.find_element(By.CSS_SELECTOR, '.smv__verticalSections.section')
    incidents = incidents_section.find_elements(By.CSS_SELECTOR, '.smv__incident')
    for inc in incidents:
        try:
            smv_assist = inc.find_element(By.CSS_SELECTOR, '.smv__assist')
            if smv_assist.text == "(Not on pitch)":
                icon = inc.find_element(By.CSS_SELECTOR, '.smv__incidentIcon')
                if "yellow card" in icon.get_attribute("title") or len(
                        driver.find_elements(By.CSS_SELECTOR, '.card-ico.yellowCard-ico')) > 0:
                    yellows_not_on_pitch_counter += 1
                    print("_____Found YELLOW not on pitch_____")
                if "red card" in icon.get_attribute("title") or len(
                        driver.find_elements(By.CSS_SELECTOR, '.card-ico.redCard-ico')) > 0:
                    reds_not_on_pitch_counter += 1
                    print("_____Found RED not on pitch_____")
        except NoSuchElementException:
            pass
            # print("_____no sub_incident found_____")

    # --- STATS ---
    Wait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//button[text()='Stats']")))
    driver.find_element(By.XPATH, "//button[text()='Stats']").click()

    Wait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.stat__row')))
    stat_rows = driver.find_elements(By.CSS_SELECTOR, '.stat__row')
    for sr in stat_rows:
        cat = sr.find_element(By.CSS_SELECTOR, '.stat__category')
        cat_name = cat.find_element(By.CSS_SELECTOR, '.stat__categoryName').text

        if cat_name == 'Ball Possession':
            new_match.possession_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text[:-1])
            new_match.possession_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text[:-1])
        elif cat_name == 'Goal Attempts':
            new_match.shots_total_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.shots_total_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Shots on Goal':
            new_match.shots_on_goal_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.shots_on_goal_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Shots off Goal':
            new_match.shots_off_goal_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.shots_off_goal_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Blocked Shots':
            new_match.shots_blocked_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.shots_blocked_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Free Kicks':
            new_match.free_kicks_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.free_kicks_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Corner Kicks':
            new_match.corner_kicks_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.corner_kicks_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Offsides':
            new_match.offsides_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.offsides_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Throw-ins':
            new_match.throw_ins_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.throw_ins_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Goalkeeper Saves':
            new_match.goalkeeper_saves_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.goalkeeper_saves_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Fouls':
            new_match.fouls_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.fouls_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Red Cards':
            new_match.red_cards_on_pitch_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.red_cards_on_pitch_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Yellow Cards':
            new_match.yellow_cards_on_pitch_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.yellow_cards_on_pitch_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Attacks':
            new_match.attacks_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.attacks_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Dangerous Attacks':
            new_match.dangerous_attacks_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.dangerous_attacks_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        else:
            raise ValueError('Unknown category name in statistics found.')

    # --- 1st HALF STATS ---
    Wait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//button[text()='1st Half']")))
    driver.find_element(By.XPATH, "//button[text()='1st Half']").click()

    Wait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.stat__row')))
    stat_rows = driver.find_elements(By.CSS_SELECTOR, '.stat__row')

    for sr in stat_rows:
        cat = sr.find_element(By.CSS_SELECTOR, '.stat__category')
        cat_name = cat.find_element(By.CSS_SELECTOR, '.stat__categoryName').text

        if cat_name == 'Ball Possession':
            new_match.possession_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text[:-1])
            new_match.possession_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text[:-1])
        elif cat_name == 'Goal Attempts':
            new_match.shots_total_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.shots_total_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Shots on Goal':
            new_match.shots_on_goal_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.shots_on_goal_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Shots off Goal':
            new_match.shots_off_goal_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.shots_off_goal_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Blocked Shots':
            new_match.shots_blocked_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.shots_blocked_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Free Kicks':
            new_match.free_kicks_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.free_kicks_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Corner Kicks':
            new_match.corner_kicks_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.corner_kicks_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Offsides':
            new_match.offsides_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.offsides_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Throw-ins':
            new_match.throw_ins_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.throw_ins_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Goalkeeper Saves':
            new_match.goalkeeper_saves_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.goalkeeper_saves_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Fouls':
            new_match.fouls_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.fouls_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Red Cards':
            new_match.red_cards_on_pitch_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.red_cards_on_pitch_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Yellow Cards':
            new_match.yellow_cards_on_pitch_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.yellow_cards_on_pitch_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Attacks':
            new_match.attacks_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.attacks_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Dangerous Attacks':
            new_match.dangerous_attacks_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.dangerous_attacks_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        else:
            raise ValueError('Unknown category name in statistics found.')

    if None in (new_match.shots_on_goal_home_1h, new_match.shots_on_goal_away_1h, new_match.goalkeeper_saves_home_1h,
                new_match.goalkeeper_saves_away_1h):
        raise ValueError('Missing statistics for 1st half: Shots on goal/Goalkeeper saves.')
    new_match.goals_home_1h = new_match.shots_on_goal_home_1h - new_match.goalkeeper_saves_away_1h
    new_match.goals_away_1h = new_match.shots_on_goal_away_1h - new_match.goalkeeper_saves_home_1h

    # --- 2nd HALF STATS ---
    Wait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//button[text()='2nd Half']")))
    driver.find_element(By.XPATH, "//button[text()='2nd Half']").click()

    Wait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.stat__row')))
    stat_rows = driver.find_elements(By.CSS_SELECTOR, '.stat__row')

    for sr in stat_rows:
        cat = sr.find_element(By.CSS_SELECTOR, '.stat__category')
        cat_name = cat.find_element(By.CSS_SELECTOR, '.stat__categoryName').text

        if cat_name == 'Ball Possession':
            new_match.possession_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text[:-1])
            new_match.possession_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text[:-1])
        elif cat_name == 'Goal Attempts':
            new_match.shots_total_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.shots_total_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Shots on Goal':
            new_match.shots_on_goal_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.shots_on_goal_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Shots off Goal':
            new_match.shots_off_goal_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.shots_off_goal_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Blocked Shots':
            new_match.shots_blocked_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.shots_blocked_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Free Kicks':
            new_match.free_kicks_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.free_kicks_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Corner Kicks':
            new_match.corner_kicks_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.corner_kicks_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Offsides':
            new_match.offsides_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.offsides_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Throw-ins':
            new_match.throw_ins_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.throw_ins_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Goalkeeper Saves':
            new_match.goalkeeper_saves_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.goalkeeper_saves_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Fouls':
            new_match.fouls_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.fouls_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Red Cards':
            new_match.red_cards_on_pitch_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.red_cards_on_pitch_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Yellow Cards':
            new_match.yellow_cards_on_pitch_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.yellow_cards_on_pitch_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Attacks':
            new_match.attacks_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.attacks_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        elif cat_name == 'Dangerous Attacks':
            new_match.dangerous_attacks_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
            new_match.dangerous_attacks_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
        else:
            raise ValueError('Unknown category name in statistics found.')

    if None in (new_match.shots_on_goal_home_2h, new_match.shots_on_goal_away_2h, new_match.goalkeeper_saves_home_2h,
                new_match.goalkeeper_saves_away_2h):
        raise ValueError('Missing statistics for 2nd half: Shots on goal/Goalkeeper saves.')
    new_match.goals_home_2h = new_match.shots_on_goal_home_2h - new_match.goalkeeper_saves_away_2h
    new_match.goals_away_2h = new_match.shots_on_goal_away_2h - new_match.goalkeeper_saves_home_2h

    return new_match


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
for match in matches[106:120]:
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
        new_match = get_match_statistics()
        list_of_matches.append(new_match)

    driver.close()
    driver.switch_to.window(driver.window_handles[0])

df = pd.DataFrame([match.to_dict() for match in list_of_matches])

break_point = 0
