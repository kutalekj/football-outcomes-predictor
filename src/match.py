from datetime import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as Wait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException


class Match:
    def __init__(self):
        self.id = None
        self.date_time = None
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

        self.referee = None
        self.neutral_field = None
        self.finished = None
        self.no_spectators = None

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

        self.free_kicks_home = -1
        self.free_kicks_away = -1
        self.corner_kicks_home = -1
        self.corner_kicks_away = -1
        self.offsides_home = -1
        self.offsides_away = -1
        self.throw_ins_home = -1
        self.throw_ins_away = -1
        self.goalkeeper_saves_home = -1
        self.goalkeeper_saves_away = -1

        self.fouls_home = -1
        self.fouls_away = -1
        self.red_cards_on_pitch_home = -1
        self.red_cards_on_pitch_away = -1
        self.yellow_cards_on_pitch_home = -1
        self.yellow_cards_on_pitch_away = -1
        self.attacks_home = -1
        self.attacks_away = -1
        self.dangerous_attacks_home = -1
        self.dangerous_attacks_away = -1
        self.total_passes_home = -1
        self.total_passes_away = -1
        self.completed_passes_home = -1
        self.completed_passes_away = -1
        self.tackles_home = -1
        self.tackles_away = -1
        self.expected_goals_home = -1
        self.expected_goals_away = -1

        self.possession_home_1h = -1
        self.possession_away_1h = -1
        self.shots_total_home_1h = -1
        self.shots_total_away_1h = -1
        self.shots_on_goal_home_1h = -1
        self.shots_on_goal_away_1h = -1
        self.shots_off_goal_home_1h = -1
        self.shots_off_goal_away_1h = -1
        self.shots_blocked_home_1h = -1
        self.shots_blocked_away_1h = -1

        self.free_kicks_home_1h = -1
        self.free_kicks_away_1h = -1
        self.corner_kicks_home_1h = -1
        self.corner_kicks_away_1h = -1
        self.offsides_home_1h = -1
        self.offsides_away_1h = -1
        self.throw_ins_home_1h = -1
        self.throw_ins_away_1h = -1
        self.goalkeeper_saves_home_1h = -1
        self.goalkeeper_saves_away_1h = -1

        self.fouls_home_1h = -1
        self.fouls_away_1h = -1
        self.red_cards_on_pitch_home_1h = -1
        self.red_cards_on_pitch_away_1h = -1
        self.yellow_cards_on_pitch_home_1h = -1
        self.yellow_cards_on_pitch_away_1h = -1
        self.attacks_home_1h = -1
        self.attacks_away_1h = -1
        self.dangerous_attacks_home_1h = -1
        self.dangerous_attacks_away_1h = -1
        self.total_passes_home_1h = -1
        self.total_passes_away_1h = -1
        self.completed_passes_home_1h = -1
        self.completed_passes_away_1h = -1
        self.tackles_home_1h = -1
        self.tackles_away_1h = -1
        self.expected_goals_home_1h = -1
        self.expected_goals_away_1h = -1

        self.goals_home_1h = -1
        self.goals_away_1h = -1

        self.possession_home_2h = -1
        self.possession_away_2h = -1
        self.shots_total_home_2h = -1
        self.shots_total_away_2h = -1
        self.shots_on_goal_home_2h = -1
        self.shots_on_goal_away_2h = -1
        self.shots_off_goal_home_2h = -1
        self.shots_off_goal_away_2h = -1
        self.shots_blocked_home_2h = -1
        self.shots_blocked_away_2h = -1

        self.free_kicks_home_2h = -1
        self.free_kicks_away_2h = -1
        self.corner_kicks_home_2h = -1
        self.corner_kicks_away_2h = -1
        self.offsides_home_2h = -1
        self.offsides_away_2h = -1
        self.throw_ins_home_2h = -1
        self.throw_ins_away_2h = -1
        self.goalkeeper_saves_home_2h = -1
        self.goalkeeper_saves_away_2h = -1

        self.fouls_home_2h = -1
        self.fouls_away_2h = -1
        self.red_cards_on_pitch_home_2h = -1
        self.red_cards_on_pitch_away_2h = -1
        self.yellow_cards_on_pitch_home_2h = -1
        self.yellow_cards_on_pitch_away_2h = -1
        self.attacks_home_2h = -1
        self.attacks_away_2h = -1
        self.dangerous_attacks_home_2h = -1
        self.dangerous_attacks_away_2h = -1
        self.total_passes_home_2h = -1
        self.total_passes_away_2h = -1
        self.completed_passes_home_2h = -1
        self.completed_passes_away_2h = -1
        self.tackles_home_2h = -1
        self.tackles_away_2h = -1
        self.expected_goals_home_2h = -1
        self.expected_goals_away_2h = -1

        self.goals_home_2h = -1
        self.goals_away_2h = -1

    def to_dict(self):
        return {
            'id': self.id,
            'date_time': self.date_time,
            'team_home': self.team_home,
            'team_away': self.team_away,
            'goals_home': self.goals_home,
            'goals_away': self.goals_away,
            'result': self.result,
            'country': self.country,
            'competition': self.competition,
            'season': self.season,
            'round': self.round,
            'referee': self.referee,
            'neutral_field': self.neutral_field,
            'finished': self.finished,
            'no_spectators': self.no_spectators,
            'odd_tipsport_1_start': self.odd_tipsport_1_start,
            'odd_tipsport_1_end': self.odd_tipsport_1_end,
            'odd_tipsport_0_start': self.odd_tipsport_0_start,
            'odd_tipsport_0_end': self.odd_tipsport_0_end,
            'odd_tipsport_2_start': self.odd_tipsport_2_start,
            'odd_tipsport_2_end': self.odd_tipsport_2_end,
            'odd_fortuna_1_start': self.odd_fortuna_1_start,
            'odd_fortuna_1_end': self.odd_fortuna_1_end,
            'odd_fortuna_0_start': self.odd_fortuna_0_start,
            'odd_fortuna_0_end': self.odd_fortuna_0_end,
            'odd_fortuna_2_start': self.odd_fortuna_2_start,
            'odd_fortuna_2_end': self.odd_fortuna_2_end,
            'possession_home': self.possession_home,
            'possession_away': self.possession_away,
            'shots_total_home': self.shots_total_home,
            'shots_total_away': self.shots_total_away,
            'shots_on_goal_home': self.shots_on_goal_home,
            'shots_on_goal_away': self.shots_on_goal_away,
            'shots_off_goal_home': self.shots_off_goal_home,
            'shots_off_goal_away': self.shots_off_goal_away,
            'shots_blocked_home': self.shots_blocked_home,
            'shots_blocked_away': self.shots_blocked_away,
            'free_kicks_home': self.free_kicks_home,
            'free_kicks_away': self.free_kicks_away,
            'corner_kicks_home': self.corner_kicks_home,
            'corner_kicks_away': self.corner_kicks_away,
            'offsides_home': self.offsides_home,
            'offsides_away': self.offsides_away,
            'throw_ins_home': self.throw_ins_home,
            'throw_ins_away': self.throw_ins_away,
            'goalkeeper_saves_home': self.goalkeeper_saves_home,
            'goalkeeper_saves_away': self.goalkeeper_saves_away,
            'fouls_home': self.fouls_home,
            'fouls_away': self.fouls_away,
            'red_cards_on_pitch_home': self.red_cards_on_pitch_home,
            'red_cards_on_pitch_away': self.red_cards_on_pitch_away,
            'yellow_cards_on_pitch_home': self.yellow_cards_on_pitch_home,
            'yellow_cards_on_pitch_away': self.yellow_cards_on_pitch_away,
            'attacks_home': self.attacks_home,
            'attacks_away': self.attacks_away,
            'dangerous_attacks_home': self.dangerous_attacks_home,
            'dangerous_attacks_away': self.dangerous_attacks_away,
            'tackles_home': self.tackles_home,
            'tackles_away': self.tackles_away,
            'total_passes_home': self.total_passes_home,
            'total_passes_away': self.total_passes_away,
            'completed_passes_home': self.completed_passes_home,
            'completed_passes_away': self.completed_passes_away,
            'expected_goals_home': self.expected_goals_home,
            'expected_goals_away': self.expected_goals_away,
            'possession_home_1h': self.possession_home_1h,
            'possession_away_1h': self.possession_away_1h,
            'shots_total_home_1h': self.shots_total_home_1h,
            'shots_total_away_1h': self.shots_total_away_1h,
            'shots_on_goal_home_1h': self.shots_on_goal_home_1h,
            'shots_on_goal_away_1h': self.shots_on_goal_away_1h,
            'shots_off_goal_home_1h': self.shots_off_goal_home_1h,
            'shots_off_goal_away_1h': self.shots_off_goal_away_1h,
            'shots_blocked_home_1h': self.shots_blocked_home_1h,
            'shots_blocked_away_1h': self.shots_blocked_away_1h,
            'free_kicks_home_1h': self.free_kicks_home_1h,
            'free_kicks_away_1h': self.free_kicks_away_1h,
            'corner_kicks_home_1h': self.corner_kicks_home_1h,
            'corner_kicks_away_1h': self.corner_kicks_away_1h,
            'offsides_home_1h': self.offsides_home_1h,
            'offsides_away_1h': self.offsides_away_1h,
            'throw_ins_home_1h': self.throw_ins_home_1h,
            'throw_ins_away_1h': self.throw_ins_away_1h,
            'goalkeeper_saves_home_1h': self.goalkeeper_saves_home_1h,
            'goalkeeper_saves_away_1h': self.goalkeeper_saves_away_1h,
            'fouls_home_1h': self.fouls_home_1h,
            'fouls_away_1h': self.fouls_away_1h,
            'red_cards_on_pitch_home_1h': self.red_cards_on_pitch_home_1h,
            'red_cards_on_pitch_away_1h': self.red_cards_on_pitch_away_1h,
            'yellow_cards_on_pitch_home_1h': self.yellow_cards_on_pitch_home_1h,
            'yellow_cards_on_pitch_away_1h': self.yellow_cards_on_pitch_away_1h,
            'attacks_home_1h': self.attacks_home_1h,
            'attacks_away_1h': self.attacks_away_1h,
            'dangerous_attacks_home_1h': self.dangerous_attacks_home_1h,
            'dangerous_attacks_away_1h': self.dangerous_attacks_away_1h,
            'tackles_home_1h': self.tackles_home_1h,
            'tackles_away_1h': self.tackles_away_1h,
            'total_passes_home_1h': self.total_passes_home_1h,
            'total_passes_away_1h': self.total_passes_away_1h,
            'completed_passes_home_1h': self.completed_passes_home_1h,
            'completed_passes_away_1h': self.completed_passes_away_1h,
            'expected_goals_home_1h': self.expected_goals_home_1h,
            'expected_goals_away_1h': self.expected_goals_away_1h,
            'goals_home_1h': self.goals_home_1h,
            'goals_away_1h': self.goals_away_1h,
            'possession_home_2h': self.possession_home_2h,
            'possession_away_2h': self.possession_away_2h,
            'shots_total_home_2h': self.shots_total_home_2h,
            'shots_total_away_2h': self.shots_total_away_2h,
            'shots_on_goal_home_2h': self.shots_on_goal_home_2h,
            'shots_on_goal_away_2h': self.shots_on_goal_away_2h,
            'shots_off_goal_home_2h': self.shots_off_goal_home_2h,
            'shots_off_goal_away_2h': self.shots_off_goal_away_2h,
            'shots_blocked_home_2h': self.shots_blocked_home_2h,
            'shots_blocked_away_2h': self.shots_blocked_away_2h,
            'free_kicks_home_2h': self.free_kicks_home_2h,
            'free_kicks_away_2h': self.free_kicks_away_2h,
            'corner_kicks_home_2h': self.corner_kicks_home_2h,
            'corner_kicks_away_2h': self.corner_kicks_away_2h,
            'offsides_home_2h': self.offsides_home_2h,
            'offsides_away_2h': self.offsides_away_2h,
            'throw_ins_home_2h': self.throw_ins_home_2h,
            'throw_ins_away_2h': self.throw_ins_away_2h,
            'goalkeeper_saves_home_2h': self.goalkeeper_saves_home_2h,
            'goalkeeper_saves_away_2h': self.goalkeeper_saves_away_2h,
            'fouls_home_2h': self.fouls_home_2h,
            'fouls_away_2h': self.fouls_away_2h,
            'red_cards_on_pitch_home_2h': self.red_cards_on_pitch_home_2h,
            'red_cards_on_pitch_away_2h': self.red_cards_on_pitch_away_2h,
            'yellow_cards_on_pitch_home_2h': self.yellow_cards_on_pitch_home_2h,
            'yellow_cards_on_pitch_away_2h': self.yellow_cards_on_pitch_away_2h,
            'attacks_home_2h': self.attacks_home_2h,
            'attacks_away_2h': self.attacks_away_2h,
            'dangerous_attacks_home_2h': self.dangerous_attacks_home_2h,
            'dangerous_attacks_away_2h': self.dangerous_attacks_away_2h,
            'tackles_home_2h': self.tackles_home_2h,
            'tackles_away_2h': self.tackles_away_2h,
            'total_passes_home_2h': self.total_passes_home_2h,
            'total_passes_away_2h': self.total_passes_away_2h,
            'completed_passes_home_2h': self.completed_passes_home_2h,
            'completed_passes_away_2h': self.completed_passes_away_2h,
            'expected_goals_home_2h': self.expected_goals_home_2h,
            'expected_goals_away_2h': self.expected_goals_away_2h,
            'goals_home_2h': self.goals_home_2h,
            'goals_away_2h': self.goals_away_2h
        }

    def get_match_statistics(self, driver, coutry, comp_name, season):
        # 1. Date & Time
        date_time = driver.find_element(By.CSS_SELECTOR, '.duelParticipant__startTime > div').text
        date_time_parsed = datetime.strptime(date_time, "%d.%m.%Y %H:%M")
        self.date_time = date_time_parsed.strftime("%Y-%m-%d %H:%M")

        # 2./3. Teams
        self.team_home = driver.find_element(By.CSS_SELECTOR,
                                             '.duelParticipant__home .participant__participantName.participant__overflow > a').text
        self.team_away = driver.find_element(By.CSS_SELECTOR,
                                             '.duelParticipant__away .participant__participantName.participant__overflow > a').text
        # print(self.team_home + " - " + self.team_away)

        # 0. PK
        self.id = date_time + "_" + self.team_home + "_" + self.team_away

        # 4./5./6. Competition, Season, Round + COUNTRY
        self.country = coutry
        # self.competition = "FORTUNA:LIGA"
        self.competition = comp_name
        # self.season = "2022-2023"
        self.season = season
        self.round = int(
            driver.find_element(By.CSS_SELECTOR, '.tournamentHeader__country > a').text.split('ROUND ')[1])
        # print(self.competition + " " + self.season + ": Round " + str(self.round))

        # 7./8./9. Result, Team Goals - Home/Away
        score_div = driver.find_element(By.CSS_SELECTOR, '.detailScore__wrapper')
        score_spans = score_div.find_elements(By.TAG_NAME, 'span')

        self.goals_home = int(score_spans[0].text)
        self.goals_away = int(score_spans[2].text)

        if self.goals_home >= self.goals_away:
            if self.goals_home > self.goals_away:
                self.result = 0
            else:
                self.result = 1
        else:
            self.result = 2
        # print(str(self.goals_home) + ":" + str(self.goals_away) + "\t(winner = " + str(self.result) + ")")

        # 10. Referee
        referee_div = driver.find_element(By.CSS_SELECTOR, '.section .mi__data')
        referee_div2 = referee_div.find_elements(By.TAG_NAME, 'div')[0]
        self.referee = referee_div2.find_element(By.CSS_SELECTOR, '.mi__item__val').text.strip()

        # 11./12. Neutral field, Finished + NO_SPECTATORS?
        try:
            match_info = driver.find_element(By.CSS_SELECTOR, '.infoBox__wrapper .infoBox__info').text
            if "at a different stadium" in match_info:
                self.neutral_field = True
            else:
                self.neutral_field = False
                if "No spectators" not in match_info:
                    print("\t\t\tNEW_MATCH_INFO: " + match_info)
        except NoSuchElementException:
            self.neutral_field = False

        try:
            match_info = driver.find_element(By.CSS_SELECTOR, '.infoBox__wrapper .infoBox__info').text
            if "No spectators" in match_info:
                self.no_spectators = True
            else:
                self.no_spectators = False
                if "at a different stadium" not in match_info:
                    print("\t\t\tNEW_MATCH_INFO: " + match_info)
        except NoSuchElementException:
            self.no_spectators = False

        finished_elem = Wait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "fixedHeaderDuel__detailStatus")))
        finished_text = driver.execute_script("return arguments[0].innerText;", finished_elem)
        self.finished = True if finished_text == "FINISHED" else False
        if not self.finished:
            raise Exception("All matches must be finished")
        # print("Referee: " + self.referee + ", neutral field = " + str(self.neutral_field) + ", finished = " + str(self.finished))

        # 13.- 24. Odds (Tipsport, Fortuna)
        odds_elems = driver.find_elements(By.CSS_SELECTOR, '.oddsRowContent')
        for odd_elem, i in zip(odds_elems, range(len(odds_elems))):
            last_minute_odds = odd_elem.find_elements(By.CSS_SELECTOR, '.oddsValueInner')

            if len(last_minute_odds) < 3:
                last_minute_odd1 = -1
                last_minute_odd0 = -1
                last_minute_odd2 = -1
            else:
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

            # print(str(init_odd1) + " >> " + str(last_minute_odd1) + ", " + str(init_odd0) + " >> " + str(last_minute_odd0) + ", " + str(init_odd2) + " >> " + str(last_minute_odd2))

            if i == 0:
                self.odd_tipsport_1_start = float(init_odd1)
                self.odd_tipsport_1_end = float(last_minute_odd1)
                self.odd_tipsport_0_start = float(init_odd0)
                self.odd_tipsport_0_end = float(last_minute_odd0)
                self.odd_tipsport_2_start = float(init_odd2)
                self.odd_tipsport_2_end = float(last_minute_odd2)
            else:
                self.odd_fortuna_1_start = float(init_odd1)
                self.odd_fortuna_1_end = float(last_minute_odd1)
                self.odd_fortuna_0_start = float(init_odd0)
                self.odd_fortuna_0_end = float(last_minute_odd0)
                self.odd_fortuna_2_start = float(init_odd2)
                self.odd_fortuna_2_end = float(last_minute_odd2)

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
        try:
            Wait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//button[text()='Stats']")))
            driver.find_element(By.XPATH, "//button[text()='Stats']").click()
        except (TimeoutException, NoSuchElementException):
            return

        Wait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.stat__row')))
        stat_rows = driver.find_elements(By.CSS_SELECTOR, '.stat__row')
        for sr in stat_rows:
            cat = sr.find_element(By.CSS_SELECTOR, '.stat__category')
            cat_name = cat.find_element(By.CSS_SELECTOR, '.stat__categoryName').text

            if cat_name == 'Ball Possession':
                self.possession_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text[:-1])
                self.possession_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text[:-1])
            elif cat_name == 'Goal Attempts':
                self.shots_total_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.shots_total_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Shots on Goal':
                self.shots_on_goal_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.shots_on_goal_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Shots off Goal':
                self.shots_off_goal_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.shots_off_goal_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Blocked Shots':
                self.shots_blocked_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.shots_blocked_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Free Kicks':
                self.free_kicks_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.free_kicks_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Corner Kicks':
                self.corner_kicks_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.corner_kicks_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Offsides':
                self.offsides_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.offsides_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Throw-ins':
                self.throw_ins_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.throw_ins_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Goalkeeper Saves':
                self.goalkeeper_saves_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.goalkeeper_saves_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Fouls':
                self.fouls_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.fouls_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Red Cards':
                self.red_cards_on_pitch_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.red_cards_on_pitch_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Yellow Cards':
                self.yellow_cards_on_pitch_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.yellow_cards_on_pitch_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Attacks':
                self.attacks_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.attacks_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Dangerous Attacks':
                self.dangerous_attacks_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.dangerous_attacks_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Tackles':
                self.tackles_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.tackles_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Total Passes':
                self.total_passes_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.total_passes_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Completed Passes':
                self.completed_passes_home = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.completed_passes_away = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Expected Goals (xG)':
                self.expected_goals_home = float(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.expected_goals_away = float(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
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
                self.possession_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text[:-1])
                self.possession_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text[:-1])
            elif cat_name == 'Goal Attempts':
                self.shots_total_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.shots_total_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Shots on Goal':
                self.shots_on_goal_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.shots_on_goal_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Shots off Goal':
                self.shots_off_goal_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.shots_off_goal_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Blocked Shots':
                self.shots_blocked_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.shots_blocked_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Free Kicks':
                self.free_kicks_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.free_kicks_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Corner Kicks':
                self.corner_kicks_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.corner_kicks_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Offsides':
                self.offsides_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.offsides_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Throw-ins':
                self.throw_ins_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.throw_ins_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Goalkeeper Saves':
                self.goalkeeper_saves_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.goalkeeper_saves_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Fouls':
                self.fouls_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.fouls_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Red Cards':
                self.red_cards_on_pitch_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.red_cards_on_pitch_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Yellow Cards':
                self.yellow_cards_on_pitch_home_1h = int(
                    cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.yellow_cards_on_pitch_away_1h = int(
                    cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Attacks':
                self.attacks_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.attacks_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Dangerous Attacks':
                self.dangerous_attacks_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.dangerous_attacks_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Tackles':
                self.tackles_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.tackles_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Total Passes':
                self.total_passes_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.total_passes_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Completed Passes':
                self.completed_passes_home_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.completed_passes_away_1h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Expected Goals (xG)':
                self.expected_goals_home_1h = float(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.expected_goals_away_1h = float(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            else:
                raise ValueError('Unknown category name in statistics found.')

        if None in (
                self.shots_on_goal_home_1h, self.shots_on_goal_away_1h, self.goalkeeper_saves_home_1h,
                self.goalkeeper_saves_away_1h):
            raise ValueError('Missing statistics for 1st half: Shots on goal/Goalkeeper saves.')
        self.goals_home_1h = self.shots_on_goal_home_1h - self.goalkeeper_saves_away_1h
        self.goals_away_1h = self.shots_on_goal_away_1h - self.goalkeeper_saves_home_1h

        # --- 2nd HALF STATS ---
        Wait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//button[text()='2nd Half']")))
        driver.find_element(By.XPATH, "//button[text()='2nd Half']").click()

        Wait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.stat__row')))
        stat_rows = driver.find_elements(By.CSS_SELECTOR, '.stat__row')

        for sr in stat_rows:
            cat = sr.find_element(By.CSS_SELECTOR, '.stat__category')
            cat_name = cat.find_element(By.CSS_SELECTOR, '.stat__categoryName').text

            if cat_name == 'Ball Possession':
                self.possession_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text[:-1])
                self.possession_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text[:-1])
            elif cat_name == 'Goal Attempts':
                self.shots_total_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.shots_total_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Shots on Goal':
                self.shots_on_goal_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.shots_on_goal_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Shots off Goal':
                self.shots_off_goal_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.shots_off_goal_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Blocked Shots':
                self.shots_blocked_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.shots_blocked_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Free Kicks':
                self.free_kicks_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.free_kicks_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Corner Kicks':
                self.corner_kicks_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.corner_kicks_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Offsides':
                self.offsides_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.offsides_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Throw-ins':
                self.throw_ins_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.throw_ins_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Goalkeeper Saves':
                self.goalkeeper_saves_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.goalkeeper_saves_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Fouls':
                self.fouls_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.fouls_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Red Cards':
                self.red_cards_on_pitch_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.red_cards_on_pitch_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Yellow Cards':
                self.yellow_cards_on_pitch_home_2h = int(
                    cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.yellow_cards_on_pitch_away_2h = int(
                    cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Attacks':
                self.attacks_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.attacks_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Dangerous Attacks':
                self.dangerous_attacks_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.dangerous_attacks_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Tackles':
                self.tackles_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.tackles_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Total Passes':
                self.total_passes_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.total_passes_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Completed Passes':
                self.completed_passes_home_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.completed_passes_away_2h = int(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            elif cat_name == 'Expected Goals (xG)':
                self.expected_goals_home_2h = float(cat.find_element(By.CSS_SELECTOR, '.stat__homeValue').text)
                self.expected_goals_away_2h = float(cat.find_element(By.CSS_SELECTOR, '.stat__awayValue').text)
            else:
                raise ValueError('Unknown category name in statistics found.')

        if None in (
                self.shots_on_goal_home_2h, self.shots_on_goal_away_2h, self.goalkeeper_saves_home_2h,
                self.goalkeeper_saves_away_2h):
            raise ValueError('Missing statistics for 2nd half: Shots on goal/Goalkeeper saves.')
        self.goals_home_2h = self.shots_on_goal_home_2h - self.goalkeeper_saves_away_2h
        self.goals_away_2h = self.shots_on_goal_away_2h - self.goalkeeper_saves_home_2h

    @staticmethod
    def correct_zero_values(matches):
        attributes_to_check = [
            'possession_home', 'possession_away', 'shots_total_home', 'shots_total_away',
            'shots_on_goal_home', 'shots_on_goal_away', 'shots_off_goal_home', 'shots_off_goal_away',
            'shots_blocked_home', 'shots_blocked_away', 'free_kicks_home', 'free_kicks_away',
            'corner_kicks_home', 'corner_kicks_away', 'offsides_home', 'offsides_away',
            'throw_ins_home', 'throw_ins_away', 'goalkeeper_saves_home', 'goalkeeper_saves_away',
            'fouls_home', 'fouls_away', 'red_cards_on_pitch_home', 'red_cards_on_pitch_away',
            'yellow_cards_on_pitch_home', 'yellow_cards_on_pitch_away', 'attacks_home', 'attacks_away',
            'dangerous_attacks_home', 'dangerous_attacks_away', 'tackles_home', 'tackles_away', 'total_passes_home',
            'total_passes_away', 'completed_passes_home',
            'completed_passes_away', 'expected_goals_home', 'expected_goals_away', 'possession_home_1h',
            'possession_away_1h',
            'shots_total_home_1h', 'shots_total_away_1h', 'shots_on_goal_home_1h', 'shots_on_goal_away_1h',
            'shots_off_goal_home_1h', 'shots_off_goal_away_1h', 'shots_blocked_home_1h', 'shots_blocked_away_1h',
            'free_kicks_home_1h', 'free_kicks_away_1h', 'corner_kicks_home_1h', 'corner_kicks_away_1h',
            'offsides_home_1h', 'offsides_away_1h', 'throw_ins_home_1h', 'throw_ins_away_1h',
            'goalkeeper_saves_home_1h', 'goalkeeper_saves_away_1h', 'fouls_home_1h', 'fouls_away_1h',
            'red_cards_on_pitch_home_1h', 'red_cards_on_pitch_away_1h', 'yellow_cards_on_pitch_home_1h',
            'yellow_cards_on_pitch_away_1h', 'attacks_home_1h', 'attacks_away_1h', 'dangerous_attacks_home_1h',
            'dangerous_attacks_away_1h', 'tackles_home_1h', 'tackles_away_1h', 'total_passes_home_1h',
            'total_passes_away_1h', 'completed_passes_home_1h',
            'completed_passes_away_1h', 'expected_goals_home_1h', 'expected_goals_away_1h', 'goals_home_1h',
            'goals_away_1h', 'possession_home_2h', 'possession_away_2h',
            'shots_total_home_2h', 'shots_total_away_2h', 'shots_on_goal_home_2h', 'shots_on_goal_away_2h',
            'shots_off_goal_home_2h', 'shots_off_goal_away_2h', 'shots_blocked_home_2h', 'shots_blocked_away_2h',
            'free_kicks_home_2h', 'free_kicks_away_2h', 'corner_kicks_home_2h', 'corner_kicks_away_2h',
            'offsides_home_2h', 'offsides_away_2h', 'throw_ins_home_2h', 'throw_ins_away_2h',
            'goalkeeper_saves_home_2h', 'goalkeeper_saves_away_2h', 'fouls_home_2h', 'fouls_away_2h',
            'red_cards_on_pitch_home_2h', 'red_cards_on_pitch_away_2h', 'yellow_cards_on_pitch_home_2h',
            'yellow_cards_on_pitch_away_2h', 'attacks_home_2h', 'attacks_away_2h', 'dangerous_attacks_home_2h',
            'dangerous_attacks_away_2h', 'tackles_home_2h', 'tackles_away_2h', 'total_passes_home_2h',
            'total_passes_away_2h', 'completed_passes_home_2h',
            'completed_passes_away_2h', 'expected_goals_home_2h', 'expected_goals_away_2h', 'goals_home_2h',
            'goals_away_2h'
        ]

        # For each attribute to check
        for attr in attributes_to_check:
            # If any Match object has the current attribute greater than -1
            if any(getattr(match, attr) > -1 for match in matches):
                # Set the current attribute which is -1 to 0 for all matches
                for match in matches:
                    if getattr(match, attr) == -1:
                        setattr(match, attr, 0)

    @staticmethod
    def check_num_of_matches(matches, comp):
        if comp.finished is True and len(matches) != comp.num_of_matches_expected:
            raise ValueError(f"Found {len(matches)} matches, but {comp.num_of_matches_expected} was expected.")
