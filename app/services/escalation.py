from app.services.intercom_client import IntercomClient
from app.telegram.notifications import notify_operator


async def escalate_to_operator(
    conversation_id: str,
    question: str,
    reason: str,
    is_ticket: bool,
    intercom_client: IntercomClient,
    user_name: str = "",
    user_email: str = "",
    updated_at: str = "",
):
    message = "Ваш вопрос принят. Мы обработаем его в ближайшее время и свяжемся с вами."

    if is_ticket:
        await intercom_client.send_ticket_reply(conversation_id, message)
    else:
        await intercom_client.send_conversations_reply(conversation_id, message)

    await notify_operator(
        conversation_id=conversation_id,
        reason=reason,
        text=question,
        user_name=user_name,
        user_email=user_email,
        updated_at=updated_at,
        is_ticket=is_ticket,
    )