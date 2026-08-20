import re
import json
import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger("message_filter")

CONFIG_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "non_query_phrases.json"
)

OPERATOR_KEYWORDS = {
    "оператор",
}


@lru_cache(maxsize=1)
def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.error(
            "non_query_phrases.json not found at %s — filter disabled",
            CONFIG_PATH,
        )
        return {
            "exact_phrases": [],
            "question_indicators": [],
        }

    with open(CONFIG_PATH, encoding="utf-8") as f:
        data = json.load(f)

    logger.info(
        "Non-query phrases config loaded: %d exact phrases, "
        "%d question indicators",
        len(data.get("exact_phrases", [])),
        len(data.get("question_indicators", [])),
    )

    return data


def is_non_query_message(text: str) -> bool:
    config = _load_config()

    exact_phrases = set(config.get("exact_phrases", []))
    question_indicators = config.get("question_indicators", [])

    cleaned = re.sub(
        r"<[^>]+>",
        "",
        text or "",
    ).strip().lower()

    cleaned = cleaned.strip(".,!?;:— ")

    if not cleaned:
        return True

    if cleaned in OPERATOR_KEYWORDS:
        return False

    if cleaned in exact_phrases:
        return True

    word_count = len(cleaned.split())
    has_question_mark = "?" in text

    if word_count < 2 and not has_question_mark:
        if not any(
                keyword in cleaned
                for keyword in question_indicators
        ):
            logger.debug(
                "Non-query message detected: '%s'",
                cleaned,
            )
            return True

    return False
