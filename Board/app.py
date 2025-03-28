from flask import Flask, jsonify, request, render_template
import json
import os
import csv
from datetime import datetime, timezone

app = Flask(__name__)

board = {}  # in-memory board dictionary, keyed by match_id
BOARD_QUEUE_FILE = 'board_queue_rel.json'
COLORS_FILE = 'colors.json'
RECORDS_FILE = 'records.csv'

processed_matches = set()


def load_colors():
    if os.path.exists(COLORS_FILE):
        with open(COLORS_FILE, 'r') as f:
            return {(entry["country"], entry["comp_name"]): entry["color"] for entry in json.load(f)}
    return {}


colors = load_colors()


def load_board_queue():
    if not os.path.exists(BOARD_QUEUE_FILE):
        return []
    with open(BOARD_QUEUE_FILE, 'r') as f:
        return json.load(f)  # return list of matches from board queue file


def refresh_board():
    global board
    matches = load_board_queue()
    now = datetime.now(timezone.utc)

    print(f"🔄 [DEBUG] Refreshing board... Current Board Size: {len(board)}")

    # Rebuild board with unmarked matches
    board.clear()

    # Add new matches and update existing ones
    for match in matches:
        match_id = str(match['match_id'])
        match_dt = datetime.fromisoformat(match['datetime'])

        match["color"] = colors.get((match["country"], match["comp_name"]), "white")

        if match_dt > now and match_id not in processed_matches:  # only add/keep matches upcoming and not processed yet
            board[match_id] = match
            print(f"✅ [DEBUG] Match {match_id} added to board.")

    print(f"📋 [DEBUG] Board after refresh: {list(board.keys())}")


@app.route('/refresh', methods=['GET'])
def refresh():
    refresh_board()
    return jsonify(list(board.values()))


def kelly_criterion(prob, odds, base_bet):
    b = odds - 1  # net odds multiplier
    q = 1 - prob  # probability of loss

    if b <= 0:
        return 0  # no valid bet

    fraction = (b * prob - q) / b  # Kelly fraction
    bet_amount = max(0, fraction) * base_bet  # fraction -> actual bet size
    return bet_amount


@app.route('/trigger/<match_id>', methods=['POST'])
def trigger(match_id):
    match_id = str(match_id)

    data = request.json
    odds_yes = data.get("odds_yes", None)
    odds_no = data.get("odds_no", None)
    base_bet = data.get("base_bet", 100)  # default bet = 100

    if odds_yes is None or odds_no is None or base_bet <= 0:
        return jsonify({"error": "Both odds values and base bet must be provided"}), 400

    match = board.get(match_id)
    if not match:
        return jsonify({"error": "Match not found"}), 404

    prob = match.get("prediction")
    if prob is None:
        return jsonify({"error": "Prediction not available"}), 500

    # Compute Kelly Criterion fractions for both Yes and No bets
    recommended_fraction_yes = kelly_criterion(prob, odds_yes, base_bet)
    recommended_fraction_no = kelly_criterion(1 - prob, odds_no, base_bet)

    return jsonify({
        "match_id": match_id,
        "recommended_bet_fraction_yes": recommended_fraction_yes,
        "recommended_bet_fraction_no": recommended_fraction_no
    })


@app.route('/mark/<match_id>', methods=['POST'])
def mark(match_id):
    match_id = str(match_id)
    data = request.json or {}

    # Extract odds and recommended bet from the payload (from the front end)
    odds_yes = data.get("odds_yes")
    odds_no = data.get("odds_no")
    recommended_bet_yes = data.get("recommended_bet_yes")
    recommended_bet_no = data.get("recommended_bet_no")
    base_bet = data.get("base_bet", 100)

    # Get the match details from board or board queue
    match = board.get(match_id)
    if not match:
        # If not in board, try to find it in the board queue
        for m in load_board_queue():
            if str(m['match_id']) == match_id:
                match = m
                break
    if not match:
        return jsonify({"error": f"Match {match_id} not found"}), 404

    # Compute the time remaining (in seconds) to the match start
    now = datetime.now(timezone.utc)
    match_dt = datetime.fromisoformat(match['datetime']).astimezone(timezone.utc)
    time_remaining = (match_dt - now).total_seconds()
    if time_remaining < 0:
        time_remaining = 0

    match_start_datetime_utc = match_dt.strftime("%Y-%m-%d %H:00 UTC")

    # Append to CSV records if the match_id is not already present.
    if not os.path.exists(RECORDS_FILE):
        # Write header if file does not exist
        with open(RECORDS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "match_id", "country", "comp_name", "season", "home_team", "away_team",
                "prediction", "odds_yes", "odds_no", "base_bet", "recommended_bet_yes", "recommended_bet_no",
                "time_remaining_sec", "match_start_datetime_utc"
            ])

    # Check if the match is already recorded.
    recorded = False
    with open(RECORDS_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["match_id"] == match_id:
                recorded = True
                break

    if not recorded:
        with open(RECORDS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                match_id,
                match.get("country", ""),
                match.get("comp_name", ""),
                match.get("season", ""),
                match.get("home_team", ""),
                match.get("away_team", ""),
                match.get("prediction", ""),
                odds_yes,
                odds_no,
                base_bet,
                recommended_bet_yes,
                recommended_bet_no,
                round(time_remaining),
                match_start_datetime_utc
            ])
        print(f"📑 [DEBUG] Appended record for match {match_id} to {RECORDS_FILE}")

    # If match in processed_matches, unmark it and restore it to board
    if match_id in processed_matches:
        processed_matches.discard(match_id)
        print(f"🔄 [DEBUG] Match {match_id} marked as NOT processed.")

        matches = load_board_queue()  # restore the matches from board queue file
        for match in matches:
            if str(match['match_id']) == match_id:
                board[match_id] = match  # re-add to board
                print(f"✅ [DEBUG] Match {match_id} restored to board.")
                break  # stop once the match is found

    # If match is in board, mark as processed
    elif match_id in board:
        processed_matches.add(match_id)
        print(f"✅ [DEBUG] Match {match_id} marked as processed.")

    else:
        print(f"❌ [DEBUG] Match {match_id} not found in board or processed set")
        return jsonify({"error": f"Match {match_id} not found"}), 404

    return jsonify({"match_id": match_id, "processed": match_id in processed_matches})


@app.route('/processed_matches', methods=['GET'])
def get_processed_upcoming_matches():
    now = datetime.now(timezone.utc)

    processed_upcoming_list = []
    matches = load_board_queue()

    for match in matches:
        match_id = str(match['match_id'])
        match_dt = datetime.fromisoformat(match['datetime'])

        if match_id in processed_matches and match_dt > now:
            processed_upcoming_list.append(match)

    return jsonify(processed_upcoming_list)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
