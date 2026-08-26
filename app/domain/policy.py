from __future__ import annotations

from app.domain.response_composer import GREETING, OPERATOR_FOOTER



def escalation_ack() -> str:
    return "Ваш вопрос принят. Мы обработаем его в ближайшее время и свяжемся с вами."


def rephrase_prompt() -> str:
    return "Не удалось точно понять ваш вопрос. Пожалуйста, переформулируйте его немного подробнее."


def choice_prompt(labels: list[str]) -> str:
    from app.domain.response_composer import ResponseComposer
    return ResponseComposer.choice(labels)


def oos_response() -> str:
    return (
        "Похоже, этот вопрос не относится к тем темам, которые мы можем обработать в этом чате. "
        "Пожалуйста, напишите вопрос, связанный с обучением, школами, приложением Терра, "
        "наставничеством или доступом к сервисам.\n\n"
        "Если вопрос всё же связан с нашей системой, опишите ситуацию подробнее или напишите «оператор»."
    )
