from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    POLZA_API_KEY: str = ""
    POLZA_MODEL: str = "openai/gpt-5.6-luna-pro"
    POLZA_TEMPERATURE: float = 0.0
    INTERCOM_TOKEN: str = ""
    INTERCOM_CLIENT_SECRET: str = ""
    INTERCOM_ADMIN_ID: str = ""
    INTERCOM_ESCALATION_TAG_ID: str = ""
    DRY_RUN: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_WEBHOOK_PAYLOAD: bool = False
    FETCH_FULL_CONTEXT: bool = True
    CONTEXT_FETCH_TIMEOUT: float = 5.0
    # Kept for backward-compatible configuration parsing. ticket.created with a
    # customer description is always processed; empty/system-only events are skipped.
    PROCESS_TICKET_CREATED: bool = True
    LOG_SENSITIVE_CONTENT: bool = False
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_GROUP_ID: str = ""
    DATABASE_URL: str = ""
    KB_PATH: str = "data/knowledge_base.json"
    EMBEDDING_DEVICE: str = "cpu"
    USE_RERANKER: bool = False
    # Final LLM router. "taxonomy" classifies against the full compact KB intent directory;
    # "legacy" preserves the retrieval -> Top-K -> LLM path for rollback/benchmarking.
    LLM_ENABLED: bool = True
    LLM_ROUTER_MODE: str = "taxonomy"
    # When True, TEMPLATE requires higher evidence/margin/specificity — rare auto-answers only.
    STRICT_TEMPLATE: bool = True
    WORKER_POLL_SECONDS: float = 0.5
    MAX_CLARIFICATION_ATTEMPTS: int = 2
    LLM_CONTEXT_TURNS: int = 50
    # The same .env is shared with docker-compose, which also needs POSTGRES_*
    # variables. They are not application settings and must not prevent the API
    # from starting (or tests from importing the application).
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

settings = Settings()
