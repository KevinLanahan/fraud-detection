"""
Redis Pub/Sub consumer.

Listens on the `transactions:completed` channel for events published by
Project 1 (payment-processor). Each message is a JSON-serialized
TransactionEventIn. On receipt, the fraud engine scores the transaction
and persists a FraudCase if it exceeds the threshold.

Runs as a background asyncio task started at app startup.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import AsyncSessionLocal
from app.schemas.transaction import TransactionEventIn

logger = logging.getLogger(__name__)
settings = get_settings()


class RedisConsumer:
    def __init__(self, fraud_engine, ingest_service):
        self.fraud_engine = fraud_engine
        self.ingest_service = ingest_service
        self._running = False
        self._client: aioredis.Redis | None = None

    async def start(self):
        self._running = True
        self._client = aioredis.from_url(settings.redis_url, decode_responses=True)
        logger.info("Redis consumer starting, listening on channel: %s", settings.transaction_channel)
        asyncio.create_task(self._consume_loop())

    async def stop(self):
        self._running = False
        if self._client:
            await self._client.aclose()

    async def _consume_loop(self):
        retry_delay = 2
        while self._running:
            try:
                async with self._client.pubsub() as pubsub:
                    await pubsub.subscribe(settings.transaction_channel)
                    logger.info("Subscribed to Redis channel: %s", settings.transaction_channel)
                    async for message in pubsub.listen():
                        if not self._running:
                            break
                        if message["type"] != "message":
                            continue
                        await self._handle_message(message["data"])
                        retry_delay = 2  # reset on success
            except Exception as e:
                logger.error("Redis consumer error: %s — retrying in %ds", e, retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)  # exponential backoff, cap at 60s

    async def _handle_message(self, data: str):
        try:
            payload = json.loads(data)
            tx = TransactionEventIn(**payload)
        except Exception as e:
            logger.warning("Failed to parse transaction event: %s | raw=%s", e, data)
            return

        async with AsyncSessionLocal() as db:
            try:
                await self.ingest_service.process(tx, db)
                await db.commit()
            except Exception as e:
                await db.rollback()
                logger.error("Failed to process transaction %s: %s", payload.get("transaction_id"), e)
