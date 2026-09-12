from __future__ import annotations

import re

GREETING = "Добрый день!"
OPERATOR_FOOTER = (
    "---\n"
    "Если этот ответ не помог, просто напишите «оператор» — "
    "ваш вопрос будет передан оператору."
)

# These wrappers existed in older KB revisions. Keeping the cleanup here makes
# the response layer backward-compatible even if an old entry is reintroduced.
_LEADING_GREETING_RE = re.compile(r"^\s*добрый\s+день!?(?:\s*\n)?", re.IGNORECASE)
_OPERATOR_FOOTER_RE = re.compile(
    r"\s*(?:---\s*\n)?(?:Если этот ответ не помог решить ваш вопрос,\s*"
    r"просто напишите «оператор»\.\s*Мы повторно рассмотрим ваше обращение и ответим вам в этом чате\.?|"
    r"Если этот ответ не помог,\s*просто напишите «оператор»\s*—\s*ваш вопрос будет передан оператору\.?)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def clean_business_answer(text: str | None) -> str:
    """Return only business content, stripping legacy presentation wrappers."""
    value = str(text or "").replace("\r\n", "\n").strip()
    value = _LEADING_GREETING_RE.sub("", value, count=1).strip()
    value = _OPERATOR_FOOTER_RE.sub("", value).strip()
    return value


class ResponseComposer:
    """Single owner of user-facing response composition.

    KB entries contain business answers only. Greeting/footer are presentation
    concerns and are added exactly once at the final-response boundary.
    """

    @staticmethod
    def final_answer(answers: list[str] | tuple[str, ...], *, include_footer: bool = True) -> str:
        cleaned = [clean_business_answer(answer) for answer in answers]
        cleaned = [answer for answer in cleaned if answer]
        if not cleaned:
            return GREETING

        parts = [GREETING, *cleaned]
        if include_footer:
            parts.extend(("", OPERATOR_FOOTER))
        return "\n\n".join(parts)

    @staticmethod
    def choice(labels: list[str]) -> str:
        options = "\n".join(f"{index}. {label}" for index, label in enumerate(labels, 1))
        return (
            f"{GREETING}\n\n"
            "Похоже, ваш вопрос ближе к одному из вариантов:\n"
            f"{options}\n\n"
            "Напишите номер варианта или кратко уточните свой вопрос. "
            "Если ни один вариант не подходит, просто напишите «оператор» — "
            "ваш запрос будет передан оператору."
        )

    @staticmethod
    def rephrase(text: str) -> str:
        return f"{GREETING}\n\n{text}"

    @staticmethod
    def oos(text: str) -> str:
        return f"{GREETING}\n\n{text}"

    @staticmethod
    def escalation_ack(text: str) -> str:
        return f"{GREETING}\n\n{text}"
