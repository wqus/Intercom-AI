from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    POLZA_API_KEY: str = ""
    POLZA_MODEL: str = "anthropic/claude-sonnet-5"
    POLZA_TEMPERATURE: float = 0.0
    INTERCOM_TOKEN: str = ""
    INTERCOM_CLIENT_SECRET: str = ""
    INTERCOM_ADMIN_ID: str = ""
    INTERCOM_ESCALATION_TAG_ID: str = ""
    DRY_RUN: bool = False
    LOG_LEVEL: str = "INFO"
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_GROUP_ID: str = ""
    DATABASE_URL: str = ""
    KB_PATH: str = "data/knowledge_base.json"
    EMBEDDING_DEVICE: str = "cpu"
    USE_RERANKER: bool = False
    # Final LLM judge for supported middle-band matches.
    LLM_ENABLED: bool = True
    # When True, TEMPLATE requires higher evidence/margin/specificity — rare auto-answers only.
    STRICT_TEMPLATE: bool = True
    WORKER_POLL_SECONDS: float = 0.5
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

settings = Settings()
