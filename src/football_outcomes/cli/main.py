import argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fop.main",
        description="Run the football-outcomes pipeline (data prep / training / etc.)."
    )
    p.add_argument("--dry-run", action="store_true", help="Do nothing, just show what would run.")
    p.add_argument("--config", type=str, help="Path to a YAML/TOML config (optional).")
    return p

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # TODO: call into your real pipeline here (e.g. functions from football_outcomes.*)
    print("Pipeline stub OK. Args:", args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
