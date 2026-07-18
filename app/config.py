from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://fraud_user:fraud_pass@localhost:5435/fraud_detection"
    sync_database_url: str = "postgresql://fraud_user:fraud_pass@localhost:5435/fraud_detection"
    redis_url: str = "redis://localhost:6380"
    transaction_channel: str = "transactions:completed"

    # Rule thresholds
    velocity_max_tx_per_hour: int = 10
    amount_threshold: float = 10000.00
    round_amount_threshold: float = 1000.00

    # ML
    ml_contamination: float = 0.05
    ml_anomaly_score_threshold: float = -0.1
    ml_model_path: str = "ml/isolation_forest.pkl"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
