import re
import sys

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <logfile> <MATCH|UNMATCHED> <reason>")
    sys.exit(1)

logfile = sys.argv[1]
status = sys.argv[2]
reason = sys.argv[3]

if status not in ("MATCH", "UNMATCHED"):
    print("Status must be MATCH or UNMATCHED")
    sys.exit(1)

score_re = re.compile(r"score=([0-9.]+)")
matches = []

status_token = f"[team_strength] {status}"
reason_token = f"reason={reason}"

with open(logfile, "r", encoding="utf-8") as f:
    for line in f:
        if status_token in line and reason_token in line:
            m = score_re.search(line)
            if m:
                score = float(m.group(1))
                matches.append((score, line.rstrip()))

# deduplicate (keep highest score per line)
unique = {}
for score, line in matches:
    if line not in unique or score > unique[line]:
        unique[line] = score

matches = [(score, line) for line, score in unique.items()]

# sorting logic
reverse = status == "UNMATCHED"
matches.sort(key=lambda x: x[0], reverse=reverse)

# build output filename from reason
clean_reason = reason.removeprefix("cache:").replace(":", "_")
output_file = f"out_{clean_reason}.txt"

# output
with open(output_file, "w", encoding="utf-8") as out:
    for _, line in matches:
        out.write(line + "\n")

print(f"Wrote {len(matches)} lines to {output_file}")
