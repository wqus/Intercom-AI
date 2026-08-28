from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    YANDEX_API_KEY: str = ""
    YANDEX_FOLDER_ID: str = ""
    YANDEX_PROMPT_ID: str = ""
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
    # Conservative production routing: do not call the LLM until it is explicitly enabled.
    LLM_ENABLED: bool = False
    # User-facing fallback for ambiguous / LLM-worthy cases.
    USER_VISIBLE_LLM: bool = False
    # Top-3 category choice is enabled for full traffic (precision-first middle mode).
    USER_VISIBLE_TOP3: bool = True
    # When True, TEMPLATE requires higher evidence/margin/specificity — rare auto-answers only.
    STRICT_TEMPLATE: bool = True
    WORKER_POLL_SECONDS: float = 0.5
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

settings = Settings()
