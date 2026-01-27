
##### `team_strength_debug_20260126_205017.log`:
793784 - 32 + (32 * 11) = **794104** (total lines)

-32 * 11 = 352 (no FS lineups for team)
793752

-1767 (FS player name empty) - `team_only_fail`
791985

-6069 (FS DOB missing - set to 1970-01-01) - `team_only_fail`
785916

-48 (FS player with no SF DOB candidates) - `no_candidates`
785868

-442 (`dob_gate_fail`)
785426

-6244 (rest from `team_only_fail`)
779182

-2304
776878 (`team_dob_fail`)

794104 - 776878 = 17226 FAILED ONES (2.17%)

`dob_gate_pass` = 100
`team_only_pass` = 42595
`team_dob_pass` = 734183


`32*11` missing FS team lineups
- track lineup-missing rate per league+season (where are the missing lineups?)

`1767` missing FS player names (`team_only_fail`)
- log (league, season, team, match_id) for blanks and see if one subset of competitions is responsible

`6069` missing FS player DOB - default to 1970-01-01 (`team_only_fail`)
- check if the `SF_MATCH_TEAM_ONLY_THRESHOLD` correctly thresholds the players (possibly make it adaptive accounting for margin between two best scores?)
- check or improve player name normalization (foreign letters and strange symbols in name)?
- restrict candidates to the single snapshot nearest to match date (or a small window) instead of all roster entries across all snapshots?

`48` no candidates (`no_candidates`)
- print the unique FS player ids+names that and check if they come from the missing SOFIFA teams from Serie B (probably caused by FS team not mapped to SOFIFA team)

`442` DOB-only candidates with too low similarity (`dob_gate_fail`)
- sample random `dob_gate_fail` lines (players) and check whether the DOB-only candidates really contain players not matching the FS one and those that are from different teams (otherwise it should be `team_dob_fail`)

`6244` team-only candidates with too low similarity (`team_only_fail`)
- check if the `SF_MATCH_TEAM_ONLY_THRESHOLD` correctly thresholds the players (possibly make it adaptive accounting for margin between two best scores?)
- check or improve player name normalization (foreign letters and strange symbols in name)?
- restrict candidates to the single snapshot nearest to match date (or a small window) instead of all roster entries across all snapshots?

`2304` team+DOB candidates with too low similarity (`team_dob_fail`)
- inspect top 100 lines with the highest scores ("near misses" that might be safe to accept with a small threshold tweak or margin rule)


1. TEAM + DOB `= 734183 + 2304 = 736487`
2. TEAM only `= 42595 + 1767 + 6069 + 6244 = 56675`
3. DOB only `= 100 + 442 = 542`
`+ 48 + 32 * 11 = 400`
TOTAL = 794104


##### `team_strength_debug_20260127_144431.log`:
793784 - 32 + (32 * 11) = **794104** (total lines)

-32 * 11 = 352 (no FS lineups for team)
793752

-48 (FS player with no SF DOB candidates) - `no_candidates`
793704

-1767 (FS player name empty) - `team_only_fail`
791937

-3055 (FS DOB missing - set to 1970-01-01) - `team_only_fail`
788882

-4188 (rest from `team_only_fail`)
784694

-442 (`dob_gate_fail`)
784252

-532 (`team_dob_fail`)
783720

794104 - 783720 = 10384 FAILED ONES (1.31%)

`dob_gate_pass` = 100
`team_only_pass` = 48017
`team_dob_pass` = 735598 + 5 ('\n' in player name) = 735603

1. TEAM + DOB `= 735603 + 532 = 736135`
2. TEAM only `= 48017 + 1767 + 3055 + 4188 = 57027`
3. DOB only `= 100 + 442 = 542`
`+ 48 + 32 * 11 = 400`
TOTAL = 794104



##### `team_strength_debug_20260127_153407.log`: (FINAL)
793784 - 32 + (32 * 11) = **794104** (total lines)

-32 * 11 = 352 (no FS lineups for team)
793752

-48 (FS player with no SF DOB candidates) - `no_candidates`
793704

-1767 (FS player name empty) - `team_only_fail`
791937

-2750 (FS DOB missing - set to 1970-01-01) - `team_only_fail`
789187

-3656 (rest from `team_only_fail`)
785531

-442 (`dob_gate_fail`)
785089

-477 (`team_dob_fail`)
784612

794104 - 784612 = 9492 FAILED ONES (1.2%)

`dob_gate_pass` = 100
`team_only_pass` = 48909
`team_dob_pass` = 735598 + 5 ('\n' in player name) = 735603

1. TEAM + DOB `= 735603 + 477 = 736080`
2. TEAM only `= 48909 + 1767 + 2750 + 3656 = 57082`
3. DOB only `= 100 + 442 = 542`
`+ 48 + 32 * 11 = 400`
TOTAL = 794104

EXAMPLES:
[team_strength] MATCH fs='Alexander Brunst-Zöllner' -> sf_id=223925 score=100.0 (2nd=-1.0) (sf_name=Alexander Brunst-Zöllner) missing=0/34 gate=dob reason=cache:dob_gate_pass
[team_strength] UNMATCHED fs='Mads Greve' dob=1989-09-12 00:00:00+00:00 sf_name=Mads Enggård score=63.6 reason=team_only_fail
[team_strength] MATCH fs='Abou Ba' -> sf_id=240694 score=70.0 (2nd=-1.0) (sf_name=Abou-Malal Ba) missing=34/34 gate=dob reason=cache:team_only_pass
[team_strength] UNMATCHED fs='Mattia Mustacchio' dob=1989-05-17 00:00:00+00:00 sf_name=None score=0.0 reason=no_candidates
[team_strength] UNMATCHED fs='Simone Corazza' dob=1991-03-22 00:00:00+00:00 sf_name=Joakim Lindner score=35.7 reason=dob_gate_fail
[team_strength] UNMATCHED fs='' dob=1970-01-01 00:00:00+00:00 sf_name=Hidemasa Morita score=0.0 reason=team_only_fail
[team_strength] UNMATCHED fs='Tyrese Omotoye' dob=1970-01-01 00:00:00+00:00 sf_name=Tyrese Demola Huxley Omotoye score=66.7 reason=team_only_fail
[team_strength] MATCH fs='Luis Suárez' -> sf_id=245158 score=100.0 (2nd=-1.0) (sf_name=Luis Javier Suárez Charris) missing=0/34 gate=dob reason=cache:team_only_pass
[team_strength] lineup=[] for team_id=10059 match=2831375 (home)
[team_strength] UNMATCHED fs='' dob=1970-01-01 00:00:00+00:00 sf_name=Fabio Borini score=0.0 reason=team_only_fail
[team_strength] MATCH fs='Alexander Brunst-Zöllner' -> sf_id=223925 score=100.0 (2nd=-1.0) (sf_name=Alexander Brunst-Zöllner) missing=0/34 gate=dob reason=cache:team_dob_pass
[team_strength] UNMATCHED fs='Mads Greve' dob=1989-09-12 00:00:00+00:00 sf_name=Mads Enggård score=63.6 reason=team_only_fail

--------------------------------------------------------------------------

##### Git bash commands on the LOG file:

```bash
LOG="team_strength_debug_20260127_153407.log"

grep -c "^\[team_strength\] lineup=\[\]" "$LOG"  # missing FS lineups

grep -E -c "^\[team_strength\] (MATCH|UNMATCHED) " "$LOG"
grep -c "^\[team_strength\] MATCH " "$LOG"
grep -c "^\[team_strength\] UNMATCHED " "$LOG"

grep -E "^\[team_strength\] (MATCH|UNMATCHED) " "$LOG" | grep -c "fs=''"  # empty FS name
grep "^\[team_strength\] UNMATCHED " "$LOG" | grep -c "fs=''"
grep -E "^\[team_strength\] (MATCH|UNMATCHED) " "$LOG" | grep "fs=''" | grep -c "dob=1970-01-01"  # empty FS name implies default DOB?
grep -E "^\[team_strength\] (MATCH|UNMATCHED) " "$LOG" | grep "fs=''" | grep -v "dob=1970-01-01" | wc -l
grep -E "^\[team_strength\] (MATCH|UNMATCHED) " "$LOG" \  # default DOB among named FS players
  | grep -v "fs=''" \
  | grep -c "dob=1970-01-01"
grep "^\[team_strength\] UNMATCHED " "$LOG" \
  | grep -v "fs=''" \
  | grep -c "dob=1970-01-01"

grep -E -c "^\[team_strength\] (MATCH|UNMATCHED) .* reason=" "$LOG"  # total lines with reason
grep -E "^\[team_strength\] (MATCH|UNMATCHED) .* reason=" "$LOG" \  # normalized reason counts
  | sed -E "s/reason=cache:/reason=/" \
  | awk '
      {
        # extract reason token
        match($0, /reason=[^ ]+/, a);
        r=a[0];
        sub("reason=","",r);
        counts[r]++
      }
      END {
        for (k in counts) print k, counts[k]
      }' \
  | sort

# normalize cache prefix, then count successes
grep -E "^\[team_strength\] (MATCH|UNMATCHED) .* reason=" "$LOG" \  # success vs. failure counts in reasons
  | sed -E "s/reason=cache:/reason=/" \
  | grep -E -c "reason=.*_pass"
# normalize cache prefix, then count failures
grep -E "^\[team_strength\] (MATCH|UNMATCHED) .* reason=" "$LOG" \
  | sed -E "s/reason=cache:/reason=/" \
  | grep -E -c "reason=(.*_fail|no_candidates|no_best|unknown_stage)"

TOF=$(grep -E "^\[team_strength\] (MATCH|UNMATCHED) .* reason=" "$LOG" \  # team_only_fail normalized
  | sed -E "s/reason=cache:/reason=/" \
  | grep -c "reason=team_only_fail"); echo $TOF
TOF_EMPTY=$(grep -E "^\[team_strength\] (MATCH|UNMATCHED) .* reason=" "$LOG" \  # team_only_fail + empty name
  | sed -E "s/reason=cache:/reason=/" \
  | grep "reason=team_only_fail" \
  | grep -c "fs=''"); echo $TOF_EMPTY
TOF_NAMED_DEF=$(grep -E "^\[team_strength\] (MATCH|UNMATCHED) .* reason=" "$LOG" \  # team_only_fail + named + default DOB
  | sed -E "s/reason=cache:/reason=/" \
  | grep "reason=team_only_fail" \
  | grep -v "fs=''" \
  | grep -c "dob=1970-01-01"); echo $TOF_NAMED_DEF
echo $((TOF - TOF_EMPTY - TOF_NAMED_DEF))  # team_only_fail remainder (named + non-default DOB)
```

##### Filtering the log file based on reasons (the folder content)

```powershell
PS C:\Users\kutalekj\PycharmProjects\football-outcomes-predictor\data\processed\logs> python tmp_printer.py team_strength_debug_20260127_153407.log UNMATCHED dob_gate_fail
Wrote 44 lines to out_dob_gate_fail.txt
PS C:\Users\kutalekj\PycharmProjects\football-outcomes-predictor\data\processed\logs> python tmp_printer.py team_strength_debug_20260127_153407.log UNMATCHED team_only_fail
Wrote 668 lines to out_team_only_fail.txt
PS C:\Users\kutalekj\PycharmProjects\football-outcomes-predictor\data\processed\logs> python tmp_printer.py team_strength_debug_20260127_153407.log UNMATCHED team_dob_fail
Wrote 19 lines to out_team_dob_fail.txt
PS C:\Users\kutalekj\PycharmProjects\football-outcomes-predictor\data\processed\logs> python tmp_printer.py team_strength_debug_20260127_153407.log MATCH dob_gate_pass
Wrote 0 lines to out_dob_gate_pass.txt
PS C:\Users\kutalekj\PycharmProjects\football-outcomes-predictor\data\processed\logs> python tmp_printer.py team_strength_debug_20260127_153407.log MATCH team_only_pass
Wrote 113 lines to out_team_only_pass.txt
PS C:\Users\kutalekj\PycharmProjects\football-outcomes-predictor\data\processed\logs> python tmp_printer.py team_strength_debug_20260127_153407.log MATCH team_dob_pass
Wrote 131 lines to out_team_dob_pass.txt
PS C:\Users\kutalekj\PycharmProjects\football-outcomes-predictor\data\processed\logs>
```