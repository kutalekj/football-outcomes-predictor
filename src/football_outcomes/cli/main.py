import argparse
import sys
from pathlib import Path
from runpy import run_path

# repo root: .../src/football_outcomes/cli/main.py -> parents[3] = project root
REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_script(rel_path: str, argv: list[str]) -> int:
    """Execute a plain .py script under the repo with argv as if run directly."""
    script = REPO_ROOT / rel_path
    sys.argv = [script.name] + argv
    run_path(str(script), run_name="__main__")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fop.main",
        description="Football Outcomes  umbrella CLI",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("pipeline", help="Run the end-to-end data/features/training pipeline")
    sub.add_parser("app", help="Run the evaluation app")

    # (tools can be added later if you want them on the CLI)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)

    if args.command == "pipeline":
        return _run_script("scripts/main_apifootball.py", rest)

    if args.command == "app":
        return _run_script("scripts/main_apifootball_app.py", rest)

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
