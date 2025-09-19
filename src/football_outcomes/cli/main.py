def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Main data+features+training pipeline")
    # TODO: add real arguments here (kept minimal for now)
    p.add_argument("--config", default=None, help="Path to a YAML/JSON config")
    args = p.parse_args(argv)

    # heavy imports AFTER parsing to keep --help fast:
    from football_outcomes.data import match  # noqa: F401
    # from football_outcomes.training import train_ann  # example use
    print("fop.main is wired up. Add your orchestration logic here.")
    return 0
