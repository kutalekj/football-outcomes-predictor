from football_outcomes.application.footystats_pipeline import (
    default_pipeline_config,
    run_pipeline,
)


def main() -> None:
    run_pipeline(default_pipeline_config())


if __name__ == "__main__":
    main()
