import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import init_db
from app.dependencies import get_ingest_service
from app.routers import ingest, cases, dashboard
from app.services.redis_consumer import RedisConsumer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

_consumer: RedisConsumer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _consumer
    logger.info("Starting fraud-detection service...")

    # Create tables
    await init_db()
    logger.info("Database initialized")

    # Train model reminder
    from app.dependencies import get_ml_scorer
    scorer = get_ml_scorer()
    if not scorer.is_ready:
        logger.warning("Run 'python ml/train.py' to enable ML scoring")

    # Start Redis consumer
    ingest_svc = get_ingest_service()
    _consumer = RedisConsumer(
        fraud_engine=ingest_svc.fraud_engine,
        ingest_service=ingest_svc,
    )
    await _consumer.start()
    logger.info("Fraud detection service ready on port 8081")

    yield

    if _consumer:
        await _consumer.stop()
    logger.info("Fraud detection service stopped")


app = FastAPI(
    title="Fraud Detection Pipeline",
    description="Real-time fraud scoring for payment transactions. Consumes Project 1 events via Redis Pub/Sub.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ingest.router)
app.include_router(cases.router)
app.include_router(dashboard.router)


@app.get("/health")
async def health():
    from app.dependencies import get_ml_scorer
    return {
        "status": "ok",
        "ml_model_loaded": get_ml_scorer().is_ready,
    }
