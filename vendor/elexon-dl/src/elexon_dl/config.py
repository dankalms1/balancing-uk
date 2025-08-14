from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    base_url: str = Field(default="https://data.elexon.co.uk/bmrs/api/v1")
    timeout_s: float = 30.0
    max_retries: int = 5
    backoff_base: float = 0.5
    backoff_cap: float = 20.0
    max_concurrency: int = 128
    rate_per_sec: float = 82.0
    user_agent: str = "elexon-dl/0.2"
    cache_enabled: bool = True
    health_url: str = "https://data.elexon.co.uk/bmrs/api/v1/health"

    model_config = {
        "env_prefix": "ELEXON_",
        "extra": "ignore",
    }
