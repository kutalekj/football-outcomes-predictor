class Match:
    def __init__(self):
        self.date_time = None
        self.date_time = None

        self.team_home = None
        self.team_away = None
        self.goals_home = None
        self.goals_away = None
        self.result = None

        self.competition = None
        self.season = None
        self.round = None

        self.referee = None
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

        self.possession_home = None
        self.possession_away = None
        self.shots_total_home = None
        self.shots_total_away = None
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

        self.goals_home_2h = -1
        self.goals_away_2h = -1

    def to_dict(self):
        return {
            'date_time': self.date_time,
            'team_home': self.team_home,
            'team_away': self.team_away,
            'goals_home': self.goals_home,
            'goals_away': self.goals_away,
            'result': self.result,
            'competition': self.competition,
            'season': self.season,
            'round': self.round,
            'referee': self.referee,
            'neutral_field': self.neutral_field,
            'finished': self.finished,
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
            'goals_home_2h': self.goals_home_2h,
            'goals_away_2h': self.goals_away_2h
        }

