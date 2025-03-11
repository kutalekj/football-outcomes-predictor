from flask import Flask, jsonify, request, render_template
import json
import os
from datetime import datetime, timezone

app = Flask(__name__)

board = {}  # in-memory board dictionary, keyed by match_id
BOARD_QUEUE_FILE = 'board_queue.json'

processed_matches = set()


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

    # Add new matches and update existing ones
    for match in matches:
        match_id = str(match['match_id'])
        match_dt = datetime.fromisoformat(match['datetime'])

        if match_dt > now and match_id not in processed_matches:  # only add/keep matches upcoming and not processed yet
            board[match_id] = match
            print(f"✅ [DEBUG] Match {match_id} added to board.")

    # Remove outdated or processed matches from board
    to_remove = [match_id for match_id, match in board.items()
                 if datetime.fromisoformat(match['datetime']) <= now or match_id in processed_matches]

    for match_id in to_remove:
        print(f"🗑 [DEBUG] Removing match {match_id} from board (Processed: {board[match_id].get('processed')})")
        board.pop(match_id, None)

    print(f"📋 [DEBUG] Board after refresh: {list(board.keys())}")


@app.route('/refresh', methods=['GET'])
def refresh():
    refresh_board()
    return jsonify(list(board.values()))


def kelly_criterion(prob, odds):
    b = odds - 1
    q = 1 - prob
    if b <= 0:
        return 0
    fraction = (b * prob - q) / b  # Kelly fraction
    return max(0, fraction)


@app.route('/trigger/<match_id>', methods=['POST'])
def trigger(match_id):
    match_id = str(match_id)

    data = request.json
    odds_yes = data.get("odds_yes", None)
    odds_no = data.get("odds_no", None)

    if odds_yes is None or odds_no is None:
        return jsonify({"error": "Both odds values must be provided"}), 400

    match = board.get(match_id)
    if not match:
        return jsonify({"error": "Match not found"}), 404

    prob = match.get("prediction")
    if prob is None:
        return jsonify({"error": "Prediction not available"}), 500

    # Compute Kelly Criterion fractions for both Yes and No bets
    recommended_fraction_yes = kelly_criterion(prob, odds_yes)
    recommended_fraction_no = kelly_criterion(1 - prob, odds_no)

    return jsonify({
        "match_id": match_id,
        "recommended_bet_fraction_yes": recommended_fraction_yes,
        "recommended_bet_fraction_no": recommended_fraction_no
    })


@app.route('/mark/<match_id>', methods=['POST'])
def mark(match_id):
    match_id = str(match_id)

    if match_id not in board:
        print(f"❌ [DEBUG] Match {match_id} not found in board")
        return jsonify({"error": f"Match {match_id} not found"}), 404

    board[match_id]['processed'] = not board[match_id].get('processed', False)

    if board[match_id]['processed']:
        processed_matches.add(match_id)
    else:
        processed_matches.discard(match_id)

    print(f"✅ [DEBUG] Match {match_id} marked as {'processed' if board[match_id]['processed'] else 'unprocessed'}")

    return jsonify({"match_id": match_id, "processed": board[match_id]['processed']})


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
