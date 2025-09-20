import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
import csv
from football_outcomes.config import settings


# Init Firebase Admin (using service account file)
cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()


def fetch_records(input_csv_path, output_csv_path):

    # Load existing stored records
    existing_records = []
    cnt = 0
    with open(input_csv_path, "r", newline="", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if any(str(v).strip() == "-1" for v in row.values()):
                continue
            else:
                existing_records.append(row)
                cnt += 1
    existing_match_ids = [x["match_id"] for x in existing_records]
    print(f"ðŸ“‘ [DEBUG] {str(cnt)} existing records loaded from {input_csv_path}")

    # Get new records from Firestore Database
    records_ref = db.collection("records")
    docs = records_ref.stream()

    for doc in docs:
        data = doc.to_dict()

        if data["match_id"] in existing_match_ids:
            continue

        # Bet placed
        data["bet_placed"] = round(float(data["recommended_bet_yes"]), 2) \
            if float(data["recommended_bet_yes"]) > 0 else round(float(data["recommended_bet_no"]), 2)
        if data["bet_placed"] <= 0:
            raise ValueError("Error when getting bet placed")

        existing_records.append(data)
        existing_match_ids.append(data["match_id"])

    # Sort by datetime (asc.)
    sorted_records = []
    cnt = 0
    for rec in existing_records:
        try:
            dt = datetime.strptime(rec["match_start_datetime_utc"], "%Y-%m-%d %H:%M UTC")  # parse datetime string
        except Exception as e:
            print(f"Error parsing datetime for record {rec.get('match_id')}: {e}")
            continue
        rec["parsed_datetime"] = dt
        sorted_records.append(rec)
        cnt += 1
    sorted_records.sort(key=lambda x: x["parsed_datetime"])
    print(f"ðŸ“‘ [DEBUG] {str(cnt)} new records successfully obtained from Firestore Database and sorted by datetime")

    # Won
    for rec in sorted_records:
        if "won" not in rec.keys():
            rec["won"] = input(
                f"Enter the won amount on match between {rec['home_team']} and {rec['away_team']} played at "
                f"{rec['match_start_datetime_utc']}: ")

    # Write to output CSV file
    cnt = 0
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "match_id",
            "country",
            "comp_name",
            "season",
            "home_team",
            "away_team",
            "prediction",
            "odds_yes",
            "odds_no",
            "base_bet",
            "recommended_bet_yes",
            "recommended_bet_no",
            "time_remaining_sec",
            "match_start_datetime_utc",
            "bet_placed",
            "won"
        ])

        for rec in sorted_records:
            writer.writerow([
                rec["match_id"],
                rec["country"],
                rec["comp_name"],
                rec["season"],
                rec["home_team"],
                rec["away_team"],
                str(rec["prediction"]),
                str(rec["odds_yes"]),
                str(rec["odds_no"]),
                str(rec["base_bet"]),
                str(rec["recommended_bet_yes"]),
                str(rec["recommended_bet_no"]),
                str(rec["time_remaining_sec"]),
                rec["match_start_datetime_utc"],
                str(rec["bet_placed"]),
                str(rec["won"])
            ])
            cnt += 1
    print(f"ðŸ“‘ [DEBUG] {str(cnt)} records dumped to {output_csv_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Save locally new records from Firestore")
    parser.add_argument("input_path", type=Path, help="Path to the input CSV file with existing records")
    parser.add_argument("output_path", type=Path, help="Path to the output CSV file")
    args = parser.parse_args()
    fetch_records(args.input_path, args.output_path)

