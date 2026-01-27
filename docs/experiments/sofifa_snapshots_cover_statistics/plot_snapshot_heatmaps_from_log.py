import re
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

# ---------- CONFIG ----------
LOG_FILE = "team_strength_debug_20260127_193104.log"
FIGSIZE = (18, 10)

# Choose metric to plot: "snapshots_used" or "delta_days"
PLOT_METRIC = "snapshots_used"
AGG = "mean"  # "mean" or "std" or "count"


# ---------- HELPERS ----------
def four_month_bucket_from_datestr(datestr: str) -> str:
    # datestr like "2024-09-12" or "2024-09-12 00:00:00+00:00"
    dt = datetime.fromisoformat(datestr.replace("Z", "+00:00"))
    m = ((dt.month - 1) // 4) * 4 + 1
    return f"{dt.year:04d}-{m:02d}"


# ---------- PARSER ----------
# Example expected fields in a MATCH line:
#   league=Denmark Superliga
#   match_dt=2024-09-12 00:00:00+00:00   (or match_date=YYYY-MM-DD)
#   snapshots_used=2
#   delta_days=-28
rx = re.compile(
    r"^\[team_strength\]\s+MATCH.*?"
    r"(?:league=(?P<league>.+?))\s+.*?"
    r"(?:match_dt=(?P<match_dt>\d{4}-\d{2}-\d{2}[^\s]*))\s+.*?"
    r"snapshots_used=(?P<snapshots_used>\d+)\s+"
    r"delta_days=(?P<delta_days>-?\d+)",
)

rows = []
with open(LOG_FILE, "r", encoding="utf-8") as f:
    for line in f:
        m = rx.search(line)
        if not m:
            continue
        league = m.group("league").strip()
        match_dt = m.group("match_dt").strip()
        period = four_month_bucket_from_datestr(match_dt)
        rows.append(
            {
                "league": league,
                "period": period,
                "snapshots_used": int(m.group("snapshots_used")),
                "delta_days": int(m.group("delta_days")),
            }
        )

if not rows:
    raise SystemExit(
        "No MATCH rows parsed. Make sure your MATCH log lines include "
        "league=<...> and match_dt=<...> and snapshots_used/delta_days."
    )

df = pd.DataFrame(rows)

# ---------- AGGREGATE ----------
group = df.groupby(["league", "period"])
agg_df = group.agg(
    snapshots_used_mean=("snapshots_used", "mean"),
    snapshots_used_std=("snapshots_used", "std"),
    delta_days_mean=("delta_days", "mean"),
    delta_days_std=("delta_days", "std"),
    count=("snapshots_used", "size"),
).reset_index()

# ---------- PIVOT ----------
if AGG == "count":
    col = "count"
else:
    col = f"{PLOT_METRIC}_{AGG}"
pivot = agg_df.pivot(index="league", columns="period", values=col).sort_index(axis=1)

# ---------- PLOT ----------
plt.figure(figsize=FIGSIZE)
im = plt.imshow(pivot.values, aspect="auto")
plt.colorbar(im, label=col)

plt.yticks(range(len(pivot.index)), pivot.index)
plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=60, ha="right")
plt.title(f"SOFIFA snapshot usage heatmap ({col})")
plt.xlabel("4-month period (YYYY-MM)")
plt.ylabel("League")
plt.tight_layout()
plt.show()
