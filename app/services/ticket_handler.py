import logging

from app.services.knowledge_base import KnowledgeBase, MatchMode
from app.services.llm_client import YandexLLMClient
from app.services.intercom_client import IntercomClient

logger = logging.getLogger("ticket_handler")


async def handle_new_ticket(
        conversation_id: str,
        question: str,
        llm_client: YandexLLMClient,
        knowledge_base: KnowledgeBase,
        is_ticket: bool = False,
        intercom_client: IntercomClient | None = None,
) -> None:
    if intercom_client is None:
        intercom_client = IntercomClient()

    logger.info("Processing ticket | id=%s | question=%s", conversation_id, question[:80])

    try:
        # Fuzzy match
        match = knowledge_base.match(question)

        logger.info(
            "KB match | mode=%s | score=%.1f | category=%s",
            match.mode,
            match.score,
            match.category,
        )

        if match.mode == MatchMode.TEMPLATE:
            logger.info("Sending template answer for category: %s", match.category)
            if is_ticket:
                await intercom_client.send_ticket_reply(conversation_id, match.template_answer)
            else:
                await intercom_client.send_conversations_reply(conversation_id, match.template_answer)
            return

        if match.mode == MatchMode.NO_MATCH or not match.relevant_entries:
            logger.info("No KB match — escalating to team")
            await intercom_client.assign_to_admin(conversation_id)
            return

        # классификация
        logger.info(
            "Sending to LLM | relevant_entries=%d | hint=%s",
            len(match.relevant_entries),
            match.category,
        )

        llm_result = await llm_client.classify(
            question=question,
            relevant_entries=match.relevant_entries,
        )

        logger.info(
            "LLM result | can_answer=%s",
            llm_result.can_answer,
        )

        if llm_result.can_answer and llm_result.answer:
            if is_ticket:
                await intercom_client.send_ticket_reply(conversation_id, llm_result.answer)
            else:
                await intercom_client.send_conversations_reply(conversation_id, llm_result.answer)
        else:
            logger.info(
                "LLM cannot answer | reason=%s — escalating",
                llm_result.reason,
            )
            await intercom_client.assign_to_admin(conversation_id)

    except Exception as e:
        logger.error(
            "Unexpected error processing ticket %s: %s — escalating",
            conversation_id,
            e,
            exc_info=True,
        )
        try:
            await intercom_client.assign_to_admin(conversation_id)
        except Exception:
            logger.error("Failed to escalate ticket %s after error", conversation_id)

    finally:
        await intercom_client.close()