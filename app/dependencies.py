from functools import lru_cache
from app.services.ml_scorer import MLScorer
from app.services.fraud_engine import FraudEngine
from app.services.ingest_service import IngestService
from app.config import get_settings

settings = get_settings()

_ml_scorer: MLScorer | None = None
_fraud_engine: FraudEngine | None = None
_ingest_service: IngestService | None = None


def get_ml_scorer() -> MLScorer:
    global _ml_scorer
    if _ml_scorer is None:
        _ml_scorer = MLScorer(settings.ml_model_path)
    return _ml_scorer


def get_fraud_engine() -> FraudEngine:
    global _fraud_engine
    if _fraud_engine is None:
        _fraud_engine = FraudEngine(get_ml_scorer())
    return _fraud_engine


def get_ingest_service() -> IngestService:
    global _ingest_service
    if _ingest_service is None:
        _ingest_service = IngestService(get_fraud_engine())
    return _ingest_service
