import asyncio
import logging

from lunit_hackathon.config import Settings
from lunit_hackathon.errors import LunitHackathonError
from lunit_hackathon.l2_client import L2Client
from lunit_hackathon.schemas import ChatMessage

logger = logging.getLogger(__name__)


async def check_connection(settings: Settings | None = None) -> str:
    resolved = settings or Settings()
    async with L2Client(resolved) as client:
        completion = await client.complete(
            messages=[
                ChatMessage(
                    role="user",
                    content="연결 확인입니다. 한국어로 'L2 연결 성공'이라고 짧게 답해주세요.",
                )
            ]
        )
    if completion.content is None:
        raise RuntimeError("L2 connection check returned no text")
    return completion.content


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        answer = asyncio.run(check_connection())
    except LunitHackathonError as error:
        logger.error("L2 connection check failed: %s", error)
        return 1
    print("L2 API connection OK")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
