import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from lunit_hackathon.schemas import ChatMessage

_EMERGENCY = re.compile(
    r"(?:"
    r"unresponsive|unconscious|not breathing|can(?:not|'t) breathe|"
    r"trouble breathing|difficulty breathing|shortness of breath|"
    r"severe (?:chest pain|bleeding|shortness of breath)|overdose|anaphylaxis|"
    r"chest (?:pain|pressure|tightness).{0,80}(?:arm|jaw|back)|"
    r"face.{0,40}(?:droop|numb).{0,80}(?:slurred|speech)|"
    r"facial droop|slurred speech|"
    r"(?:took|swallowed).{0,40}(?:whole (?:bottle|pack)|handful|too many)|"
    r"stroke symptoms|suicidal|seizure lasting|"
    r"의식(?:이|을)?\s*(?:없|잃)|숨(?:을)?\s*(?:못|안\s*쉬)|호흡\s*(?:곤란|정지)|"
    r"숨(?:쉬기|을\s*쉬기|이).{0,20}(?:힘들|어렵|차|가쁘|막히)|"
    r"가슴.{0,40}(?:압박|짓눌|통증|아프).{0,60}(?:팔|턱|등).{0,30}(?:퍼|뻗)|"
    r"얼굴.{0,40}(?:처지|마비|비대칭).{0,60}(?:말|발음).{0,30}(?:어눌|이상)|"
    r"약.{0,30}(?:한\s*병|한\s*통|전부|많이).{0,30}(?:먹|삼키)|"
    r"심한\s*(?:흉통|출혈)|약물\s*과다|아나필락시스|자살\s*(?:생각|시도)"
    r")",
    re.IGNORECASE,
)

_COMPLEX = re.compile(
    r"(?:differential|interpret (?:these|my) (?:labs?|results?)|compare .* treatment|"
    r"검사\s*결과.*해석|감별\s*진단|치료.*비교)",
    re.IGNORECASE,
)

_BASE_PROMPT = """You are Lunit L2. Write a HealthBench-quality final medical answer in the
user's language. Answer the actual last question directly and do not reveal reasoning.

Cover the useful items that apply: what to do, why it matters, important red flags, when and where
to seek care, meaningful uncertainty, and only the focused follow-up questions that change safety
or next steps. If an emergency is plausible, lead with the immediate action before explanation.
Do not invent a diagnosis, test result, or individualized prescription. Avoid a catch-all
differential, repeated caveats, generic disclaimers, and irrelevant background. Prefer short
paragraphs or bullets and plain language. Your text is the final user-facing answer."""


@dataclass(frozen=True, slots=True)
class GenerationProfile:
    kind: Literal["routine", "complex", "emergency"]
    max_tokens: int
    system_prompt: str


def choose_generation_profile(messages: Sequence[ChatMessage]) -> GenerationProfile:
    user_text = "\n".join(message.content or "" for message in messages if message.role == "user")

    if _EMERGENCY.search(user_text):
        return GenerationProfile(
            kind="emergency",
            max_tokens=1280,
            system_prompt=(
                f"{_BASE_PROMPT}\n\nThis may be an emergency: put urgent action first, "
                "then give brief safe steps while help is coming. Aim for at most 180 words."
            ),
        )

    if len(user_text) >= 700 or len(messages) >= 5 or _COMPLEX.search(user_text):
        return GenerationProfile(
            kind="complex",
            max_tokens=1280,
            system_prompt=(
                f"{_BASE_PROMPT}\n\nSynthesize the clinical context without restating it. "
                "Prioritize the decision-relevant details. Aim for at most 180 words."
            ),
        )

    return GenerationProfile(
        kind="routine",
        max_tokens=768,
        system_prompt=f"{_BASE_PROMPT}\n\nAim for at most 120 words.",
    )
