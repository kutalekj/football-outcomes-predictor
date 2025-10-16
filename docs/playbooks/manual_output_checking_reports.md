Manual output checking (4.3.2025)

1. `cls._instance.debug_logger_missing_fs_lineups = []` if no lineup found in `calculate_team_strength(...)`
[D] There were 60 teams found with missing FS lineups: [260, 266, 3563, 3575, 3573, 645, 3575, 607, 3577, 3573, 1001, 3575, 3573, 1004, 3575, 1002, 996, 3575, 3563, 3573, 3573, 607, 3575, 3589, 549, 3575, 3573, 996, 3575, 1010, 564, 3575, 1001, 3573, 3573, 3589, 3575, 3578, 549, 3573, 3574, 3575, 3573, 3574, 3575, 611, 564, 3573, 3575, 3577, 1005, 3575, 3573, 3578, 611, 3573, 3575, 645, 733, 261]

(60x missing team FS lineups in a match; 22 different teams)
60 team strength imitation from ~50 000 team presences in a match (~25 000 regular matches) = 60 / 50000 = 0.0012 = 0.12% (OK)

2. `cls._instance.debug_logger_missing_fs_dobs = []` if no `fs_player['fs_birthday'] in global_instance.sofifa_players_by_dob` in `calculate_team_strength(...)`
[D] There were 12557 players found with missing FS dobs: 
players = ['Oliver Antman', 'Jannis Heuer', 'Léon Delpech', 'Rene Swete', 'Caleb Zady Sery', 'Andreas Hountondji', 'Ousmane Camara', 'Darlin Yongwa', "Konan N'Dri", 'Jannis Heuer', 'Adam Legzdins', 'Jay Henderson', 'Rene Swete', 'Darlin Yongwa', 'Ousmane Camara', "Konan N'Dri", 'Lassine Sinayoko', 'Robert Atkinson', 'Alex Scott', 'Jed Wallace', 'Jed Wallace', 'Ethan Briston', 'Femi Azeez', 'Wes McDonald', 'Daniel Neill', 'Gassan Ahadme Yahyai', 'Oli Pendlebury', 'Sam Burns', 'Kaine Kessler', 'Oscar Threlkeld', 'Joseph Anang', 'Remeao Hutton', 'Udoka Godwin-Malife', 'Carl Rushworth', ...]

706 different players in total; top 5 occurrences: Roshan Singh: 171, Jed Wallace: 162, Remeao Hutton: 152, Daniel Neill: 148, Lukas Jensen: 132
57 different players with at least 50 DOB mismatches in FS/SOFIFA player matching.
12 557 / (25 000 * 2 * 11) = 0.0228 = 2.3% (still acceptable?)
Approx. ratio of players with at least 1 DOB mismatch = 706 / (24 comps * ~20 teams per comp * ~20 players in team FS roster) = 0.0735 = 7.4% (not good...)
Consider more advanced FS/SOFIFA players matching here.

3. `cls._instance.debug_logger_missing_csv_files_for_sf_player = []` if no available player CSV files within a timedelta range in `get_sf_player_data(...)`
[D] There were 6085 players found with missing CSV files (SF): [265518, 264670, 234877, 264670, 265518, 253224, 264710, 264446, 234877, 265518, 264710, 53110, 264670, 214906, 53110, 264670, 266251, 209532, 214906, 243847, 266577, 264229, 266096, 205939, 266660, 234877, 233588, 271497, 261360, 264670, 214906, 266577, 243847, 266251, 193333, 219655, 234747, 266872, 198223, 207553, 268462, 201313, 237669, 214906, 266577, 243847, 230858, 201914, 210981, 230646, 259058, 268683, 253071, 268703, 178117, 189288, 251930, 178510, 204061, 242411, 242204, 192853, 209532, 266096, 233425, 208146, 244229, 258564, 227223, 226267, 239838, 268875, 244546, 228882, 227918, 266660, 267963, 262525, 234877, 264670, 271497, 53110, 198633, 228513, 170565, 240217, 224123, 220084, 230072, 224977, 189118, 216799, 268862, 197926, 272458, 225610, 220933, 268673, 238474, 268674, 225037, 235328, 268744, 240694, 201299, 229750, 268573, 189751, 198079, 268576, 192365, 173132, 219983, 203721, ...]

774 different players in total; top occurrences: 233588: 107, 214906: 101, 209532: 84, 205939: 81, 226677: 81
81 different players with at least 20 occurrences of no available CSV file for a their match (within the pre-defined timedelta range).

Should be improved automatically once obtaining more SOFIFA player skills data.

4. All AF/FS team matches OK in `assign_fs_team_id_team_name_by_comp_season(...)`