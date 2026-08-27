# Bookmaker betting-utility analysis

## Scope

The extended analysis uses the historical out-of-sample Under/Over 2.5
predictions employed by the original bookmaker benchmark.

- OOS predictions with valid Under/Over odds: 27,987
- Candidate betting sides: 55,974
- One best-side candidate per match: 27,987
- Matches with non-negative best-side probability edge: 16,905

The probability edge for a candidate side is

    p_model - 1 / decimal_odds

and flat-stake ROI uses one unit per selected bet.

## Main findings

### Selectivity

Broad positive-edge betting is not profitable. Historical ROI improves
substantially when only the strongest model-bookmaker discrepancies are
retained.

The strongest positive-edge decile is profitable while the weaker
positive-edge deciles are not.

### Under versus Over

The economically useful signal is concentrated primarily in Under 2.5
selections.

At a 10-percentage-point minimum probability edge, the historical
Under-only strategy produced approximately 9% ROI over 768 bets.

### Walk-forward Under strategy

Thresholds were selected using earlier seasons only and then frozen for
the next season.

- 2022: selected edge 7%, 595 bets, ROI +7.17%
- 2023: selected edge 10%, 203 bets, ROI +12.81%
- 2024: selected edge 10%, 142 bets, ROI +2.00%

This is the most encouraging temporal result in the extended analysis.

### Odds, competition and bookmaker margin

Profitability is heterogeneous across competitions and bookmaker-margin
bands.

The strong Under result is not restricted to one narrow decimal-odds
range, although subgroup support becomes limited for highly selective
strategies.

### Estimated expected return

Ranking by model-estimated expected return reproduces the same broad
selectivity phenomenon but does not clearly outperform ranking by raw
probability edge.

### Calibration

Chronological Platt and isotonic calibration do not improve the raw
football model overall.

### Bookmaker-model hybrid

A simple chronological bookmaker/model logistic hybrid obtains a
slightly higher pooled ROC AUC than bookmaker fair probabilities alone,
but the improvement is very small and the bookmaker remains marginally
better by Brier score and log loss.

Large apparent hybrid betting ROIs at high thresholds have very small
bet counts and are not interpreted as robust evidence.

## Current interpretation

The analysis does not support indiscriminate betting with the football
model.

The most promising regime is selective Under 2.5 betting when the raw
football-model probability substantially exceeds the corresponding
bookmaker break-even probability.

The Under-only walk-forward experiment is the strongest result because
its threshold was chosen from earlier seasons and produced positive ROI
in each of the three subsequent test seasons.

Further statistical uncertainty analysis, odds provenance, execution
assumptions and genuinely untouched confirmation should be performed
before making prospective profitability claims.
