from datetime import datetime
from app.core.config import settings
from app.telegram.bot import bot


async def notify_operator(
        conversation_id: str,
        reason: str,
        text: str,
        user_name: str = "",
        user_email: str = "",
        updated_at: str = "",
        is_ticket: bool = False,
):
    ticket_type = "тикет" if is_ticket else "диалог"

    lines = [
        f"<b>Запрос пользователя был передан оператору</b>",
        "",
        f"<b>ID {'тикета' if is_ticket else 'диалога'}:</b> <code>{conversation_id}</code>",
        f"<b>Тип:</b> {ticket_type}",
        f"<b>Причина эскалации:</b> {reason}",
    ]

    if user_name:
        lines.append(f"<b>Имя:</b> {user_name}")
    else:
        lines.append(f"<b>Имя:</b> —")

    if user_email:
        lines.append(f"<b>Email:</b> {user_email}")
    else:
        lines.append(f"<b>Email:</b> —")

    if updated_at:
        try:
            dt = datetime.fromtimestamp(int(updated_at))
            lines.append(f"<b>Обновлено:</b> {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            lines.append(f"<b>Обновлено:</b> {updated_at}")
    else:
        lines.append(f"<b>Обновлено:</b> —")

    lines.append("")
    lines.append(f"<b>Текст обращения:</b>")
    lines.append(f"<blockquote>{text}</blockquote>")

    await bot.send_message(
        chat_id=settings.TELEGRAM_GROUP_ID,
        text="\n".join(lines),
        parse_mode="HTML",
    )