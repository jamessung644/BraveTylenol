import logging


def configure_logging(level_name: str = "INFO") -> None:
    """Configure concise logs that never include prompts, evidence, or credentials."""

    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("lunit_hackathon").setLevel(level)
