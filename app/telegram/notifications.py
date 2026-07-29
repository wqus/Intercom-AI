from app.core.config import settings
from app.telegram.bot import bot


async def notify_operator(user_id: int, reason: str, text: str):
    await bot.send_message(
        chat_id=settings.TELEGRAM_GROUP_ID,
        text=(
            "<b>Запрос пользователя был передан оператору</b>\n\n"
            f"<b>ID пользователя:</b> <code>{user_id}</code>\n"
            f"<b>Причина эскалации:</b> {reason}\n\n"
            f"<b>Текст обращения:</b>\n"
            f"<blockquote>{text}</blockquote>"
        ),
        parse_mode="HTML",
    )