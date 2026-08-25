from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


# The project currently boots with metadata.create_all() instead of Alembic.
# Keep the existing startup approach, but explicitly add newly introduced
# nullable analytics columns so an existing production table is upgraded too.
SUPPORT_INTERACTION_ANALYTICS_COLUMNS = {
    "semantic_score": "DOUBLE PRECISION",
    "lexical_score": "DOUBLE PRECISION",
    "fuzzy_score": "DOUBLE PRECISION",
    "evidence_score": "DOUBLE PRECISION",
    "real_intent_evidence": "DOUBLE PRECISION",
    "embedding_score": "DOUBLE PRECISION",
    "intent_fit_score": "DOUBLE PRECISION",
    "family_fit_score": "DOUBLE PRECISION",
    "domain_support_score": "DOUBLE PRECISION",
    "ontology_alignment_score": "DOUBLE PRECISION",
    "negative_contrast_score": "DOUBLE PRECISION",
    "contrastive_margin": "DOUBLE PRECISION",
    "contrastive_competitor_id": "VARCHAR(64)",
    "contrastive_reason": "VARCHAR(255)",
}


async def ensure_escalation_blocklist(conn) -> None:
    """Create/upgrade the persistent post-escalation blocklist on existing installs.

    The application uses PostgreSQL in production. Keeping this idempotent makes
    deployment safe when the database predates the blocklist table, while
    Base.metadata.create_all() remains responsible for fresh installations.
    """
    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS escalated_conversations (
            conversation_id VARCHAR(128) PRIMARY KEY,
            channel VARCHAR(32) NOT NULL,
            reason VARCHAR(64) NOT NULL,
            user_name VARCHAR(255) NOT NULL DEFAULT '',
            user_email VARCHAR(320) NOT NULL DEFAULT '',
            last_message_text TEXT NOT NULL DEFAULT '',
            last_message_at VARCHAR(64) NOT NULL DEFAULT '',
            source_event_id INTEGER NULL,
            escalated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """))
    await conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_escalated_conversations_reason "
        "ON escalated_conversations (reason)"
    ))
    await conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_escalated_conversations_source_event_id "
        "ON escalated_conversations (source_event_id)"
    ))
    await conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_escalated_conversations_channel_reason "
        "ON escalated_conversations (channel, reason)"
    ))


async def ensure_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await ensure_escalation_blocklist(conn)
        for column, sql_type in SUPPORT_INTERACTION_ANALYTICS_COLUMNS.items():
            await conn.execute(
                text(
                    f"ALTER TABLE support_interactions "
                    f"ADD COLUMN IF NOT EXISTS {column} {sql_type}"
                )
            )
