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


async def ensure_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        for column, sql_type in SUPPORT_INTERACTION_ANALYTICS_COLUMNS.items():
            await conn.execute(
                text(
                    f"ALTER TABLE support_interactions "
                    f"ADD COLUMN IF NOT EXISTS {column} {sql_type}"
                )
            )
