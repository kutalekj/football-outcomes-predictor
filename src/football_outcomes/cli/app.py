def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Evaluation app runner (BoardMobile integration)")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args(argv)

    # heavy imports AFTER parsing:
    # from football_outcomes.evaluation import something
    print(f"fop.app scaffold ready. Would start app on port {args.port}.")
    return 0
