### Missing players investigation 1 (all comps together)

mp0_all_players_involved_in_AF_FS_checking: 54912
mpX_OK_players_AF_FS_matching: 702966
mp1a_AF_lineups_missing: 1870
mp1b_FS_lineups_missing: 660
mp2_AF_FS_players_matching_potential_misses: 8740
mp3_all_players_involved_in_team_strength_calculation: 749804
mp4_team_strength_complete_lineup_imitation: 660
mp5_team_strength_DOB_missing: 18463
mp6_team_strength_FS_SF_matching: 33480
mp7_team_strength_imitated_skills_as_no_CSV_data: 2547
(mp7_SKILLS_team_strength_imitated_skills_as_no_data: 229)
mp8_team_strength_imitated_players_as_no_CSV_data: 82865 *(overcounted, max. 2x; fixed in per-comp version)*
mp9_team_strength_balancing_field_to_gk: 2935
mp9_team_strength_balancing_gk_to_def: 2213
mp9_team_strength_balancing_gk_to_mid: 68
mp9_team_strength_balancing_gk_to_att: 29


##### Details

- only regular matches: >= 32_244 *(there are xG data for 32_244 matches)*
  - 32_244 * 2 >= 64_488 teams
  - 64_488 * 11 >= 709_368 players
  - should be lower bound to `mpX_OK_players_AF_FS_matching` (=702_966) + `mp0_all_players_involved_in_AF_FS_checking` (=54_912)
  - thus, **757_878** players, **68_898** teams, **34_449** regular matches?

- AF/FS matching (total = 757_878)
  - 54_912 potentially not OK
    - 1_870 surely not OK - AF lineups missing in data source (`mp1a_AF_lineups_missing`)
    - 660 surely not OK - FS lineups missing in data source (`mp1b_FS_lineups_missing`)
    - 54_192 - (1_870 + 660) = 51_662 participated in matching *(matches between 16.10.2025 and 6.11.2025)* -> (1_870 + 660) / 757_878 = **0.3% lineups missing**
    - 8_740 potentially missmatched (`mp2_AF_FS_players_matching_potential_misses`)
      - approx. 6_376 OK and 2_364 not OK (GPT-5)
      - -> 2_364 / 51_662 ~= **4% missmatched**
    - -> 54_912 - (1_870 + 660 + 8_740) = 43_642 OK

- team strength calculation (total = 749_804 by `mp3_all_players_involved_in_team_strength_calculation`)
  - 660 missing from FS lineups (`mp1b_FS_lineups_missing` = `mp4_team_strength_complete_lineup_imitation`) - *660 excluded from operations up to `mp9`*
  - 18_463 missing from DOB matching (`mp5_team_strength_DOB_missing`), giving 18_463 / 749_804 = **DOB missing for 2.5% players**
  - 33_480 missing from FS/SF low similarity (`mp6_team_strength_FS_SF_matching`), giving 33_480 / 749_804 = **4.5% players not matching FS/SF** under current similarity threshold (55%)
  - -> 749_804 - (660 + 18_463 + 33_480) = 697_201 proceed to "get_sf_player_data(...)"
    - 2_547 missing from no CSV files found in specified range (`mp7_team_strength_imitated_skills_as_no_CSV_data`) - *2_547 imitations from precomputed averages*
  - -> >= (82_865 / 2) >= 41_432 + 2_547 = 43_979 imputations (`mp8_team_strength_imitated_players_as_no_CSV_data` and it should equal to `mp5_team_strength_DOB_missing` + `mp6_team_strength_FS_SF_matching` + `mp7_team_strength_imitated_skills_as_no_CSV_data` = 54_490)
    - 54_490 / 749_804 = **7.2% players need imputation** for some reason
  - 2_935 + 2_213 + 68 + 29 = 5_245 players balanced to have exactly 1 goalkeeper in lineup (`mp9_team_strength_balancing_field_to_gk` + `mp9_team_strength_balancing_gk_to_def` + `mp9_team_strength_balancing_gk_to_mid` + `mp9_team_strength_balancing_gk_to_att`) - *should include the 660 matches without FS lineups; might include the imputed players*
  - worst-case scenario of (54_490 + 5_245) / 749_804 = **8% players imputed**

##### Summary

- 34_449 matches -> 68_898 teams -> 757_878 players
- **0.3% lineups missing** (2_530) - data not available
- up to **~4% player names missmatched** (2_364 / 51_662) - discrepancy in AF and FS data
- **2.5% players missmatched** by date of birth (18_463) - discrepancy in FS and SF data
- **4.5% player names missmatched** (33_480) - discrepancy in FS and SF data (players with same DOB matched against each other)
- results in **7-8% imputations**
