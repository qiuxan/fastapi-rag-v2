from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"
    sqlite_path: str = "data/rag.db"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def use_openai(self) -> bool:
        return bool(self.openai_api_key)
