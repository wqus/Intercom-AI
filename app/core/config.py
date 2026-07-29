from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    YANDEX_API_KEY: str
    YANDEX_FOLDER_ID: str
    YANDEX_PROMPT_ID: str

    INTERCOM_TOKEN: str = ""
    INTERCOM_CLIENT_SECRET: str = ""
    INTERCOM_ADMIN_ID: str = ""

    DRY_RUN: bool = True

    LOG_LEVEL: str = "INFO"

    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_GROUP_ID: str = ""

    POSTGRES_DB: str = ""
    POSTGRES_USER: str = ""
    POSTGRES_PASSWORD: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()