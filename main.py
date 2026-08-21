import argparse
from collections.abc import Sequence

import uvicorn

from lunit_hackathon.config import Settings
from lunit_hackathon.live_check import main as live_check_main
from lunit_hackathon.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lunit L2 medical-chat service")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="start the OpenAI-compatible API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    subparsers.add_parser("check-l2", help="run one direct live L2 API check")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check-l2":
        return live_check_main()

    settings = Settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        "app:app",
        host=getattr(args, "host", "0.0.0.0"),
        port=getattr(args, "port", 8000),
        log_level=settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
