import matplotlib.pyplot as plt
import numpy as np
import os
plt.switch_backend('TkAgg')

# Input string
input_data = """Comp 307: 98/184 = 53.261%
Comp 88: 128/214 = 59.813%
Comp 106: 114/222 = 51.351%
Comp 79: 124/202 = 61.386%
Comp 207: 85/166 = 51.205%
Comp 144: 111/207 = 53.623%
Comp 135: 153/269 = 56.877%
Comp 61: 120/209 = 57.416%
Comp 39: 165/287 = 57.491%
Comp 140: 152/273 = 55.678%
Comp 94: 110/195 = 56.410%
Comp 188: 57/93 = 61.290%
Comp 179: 101/166 = 60.843%
Comp 119: 91/162 = 56.173%
Comp 141: 164/295 = 55.593%
Comp 218: 74/139 = 53.237%
Comp 78: 131/204 = 64.216%
Comp 323: 70/103 = 67.961%
Comp 136: 125/236 = 52.966%
Comp 40: 203/373 = 54.424%
Comp 42: 162/325 = 49.846%
Comp 41: 196/353 = 55.524%
Comp 62: 106/205 = 51.707%
Comp 203: 125/217 = 57.604%"""

# Mapping of competition IDs to names and colors
comp_mapping = {
    "Comp 141": ("Segunda División", "tomato"),
    "Comp 42": ("League Two", "dodgerblue"),
    "Comp 41": ("League One", "cornflowerblue"),
    "Comp 188": ("A-League", "silver"),
    "Comp 323": ("Indian Super League", "darkkhaki"),
    "Comp 79": ("2. Bundesliga", "goldenrod"),
    "Comp 119": ("Danish Superliga", "thistle"),
    "Comp 78": ("Bundesliga", "gold"),
    "Comp 144": ("Jupiler Pro League", "aquamarine"),
    "Comp 40": ("Championship", "royalblue"),
    "Comp 61": ("Ligue 1", "blueviolet"),
    "Comp 140": ("La Liga", "red"),
    "Comp 203": ("Süpér Lig", "rosybrown"),
    "Comp 307": ("Saudi Pro League", "darkseagreen"),
    "Comp 106": ("Ekstraklasa", "lightpink"),
    "Comp 39": ("Premier League", "blue"),
    "Comp 136": ("Serie B", "lightgreen"),
    "Comp 62": ("Ligue 2", "mediumorchid"),
    "Comp 135": ("Serie A", "limegreen"),
    "Comp 179": ("Premiership", "lavender"),
    "Comp 88": ("Eredivisie", "orange"),
    "Comp 218": ("Austrian Bundesliga", "lavenderblush"),
    "Comp 207": ("Swiss Super League", "paleturquoise"),
    "Comp 94": ("Primeira Liga", "honeydew"),
}

NAME_STRING = "RNN-all-rounds_embeds-6-4_1"
SAVE_PATH = "/src/apifootball_model/comps_val_acc"

# Parse the input data
results = []
for line in input_data.strip().split("\n"):
    parts = line.split(': ')
    comp_id = parts[0].strip()
    values = parts[1].split('/')
    correct = int(values[0].strip())
    total, ratio = values[1].split('=')
    total = int(total.strip())
    ratio = float(ratio.strip().strip('%'))
    results.append((comp_id, correct, total, ratio))

# Sort by accuracy in descending order
results = sorted(results, key=lambda x: x[3], reverse=True)

# Extract data for plotting
comp_ids, corrects, totals, ratios = zip(*results)
comp_names = [comp_mapping[comp_id][0] for comp_id in comp_ids]
colors = [comp_mapping[comp_id][1] for comp_id in comp_ids]

# Normalize total matches for bar width
normalized_totals = [total / max(totals) * 0.9 for total in totals]

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(comp_names, ratios, color=colors, height=normalized_totals)

# Add values on bars
for bar, ratio in zip(bars, ratios):
    ax.text(
        bar.get_width() + 1,  # position slightly to the right of the bar
        bar.get_y() + bar.get_height() / 2,  # vertically centered
        f"{ratio:.2f}%",
        va='center',
        ha='left',
        fontsize=10
    )

# Add total number of matches inside the bars
for bar, total in zip(bars, totals):
    ax.text(
        bar.get_width() / 2,  # center of the bar
        bar.get_y() + bar.get_height() / 2,  # vertically centered
        f"{total}",  # total matches
        va='center',
        ha='center',
        color='black',
        fontsize=9,
        fontweight='bold'
    )

# ax.set_yticks(bar_positions)
ax.set_yticklabels(comp_names, fontsize=10)
ax.invert_yaxis()  # Highest accuracy on top
# ax.set_xlabel("Total Matches", fontsize=12)
ax.set_xlabel("Accuracy [%]", fontsize=12)
ax.set_title("Validation accuracy by comp (" + NAME_STRING + ")", fontsize=14)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(SAVE_PATH, NAME_STRING) + ".png")
