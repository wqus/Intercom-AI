from app.core.config import settings
from app.telegram.bot import bot


async def notify_operator(
        conversation_id: str,
        reason: str,
        text: str,
        is_ticket: bool = False,
):
    ticket_type = "тикет" if is_ticket else "диалог"

    await bot.send_message(
        chat_id=settings.TELEGRAM_GROUP_ID,
        text=(
            f"<b>Запрос пользователя был передан оператору</b>\n\n"
            f"<b>ID {'тикета' if is_ticket else 'диалога'}:</b> <code>{conversation_id}</code>\n"
            f"<b>Тип:</b> {ticket_type}\n"
            f"<b>Причина эскалации:</b> {reason}\n\n"
            f"<b>Текст обращения:</b>\n"
            f"<blockquote>{text}</blockquote>"
        ),
        parse_mode="HTML",
    )