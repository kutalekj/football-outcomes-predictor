import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fop.app", description="Run the evaluation app (BoardMobile companion).")
    p.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # TODO: start your app here (FastAPI/Flask/Streamlit/etc.)
    print(f"App stub OK. Would start on port {args.port}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
