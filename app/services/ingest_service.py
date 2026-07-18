"""
Ingest service — orchestrates the full pipeline for a single transaction event:
  1. Persist the raw TransactionEvent
  2. Run fraud scoring (rules + ML)
  3. If flagged, persist a FraudCase with all fired rules
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.fraud_case import TransactionEvent, FraudCase, FraudRule, CaseStatus
from app.schemas.transaction import TransactionEventIn
from app.services.fraud_engine import FraudEngine

logger = logging.getLogger(__name__)


class IngestService:
    def __init__(self, fraud_engine: FraudEngine):
        self.fraud_engine = fraud_engine

    async def process(self, tx: TransactionEventIn, db: AsyncSession) -> TransactionEvent:
        # ── Idempotency: skip if already processed ───────────────────────────
        existing = await db.execute(
            select(TransactionEvent).where(TransactionEvent.transaction_id == tx.transaction_id)
        )
        if existing.scalar_one_or_none():
            logger.debug("Transaction %s already processed, skipping", tx.transaction_id)
            return None

        # ── Persist event ────────────────────────────────────────────────────
        event = TransactionEvent(
            transaction_id=tx.transaction_id,
            source_account_id=tx.source_account_id,
            dest_account_id=tx.dest_account_id,
            amount=tx.amount,
            currency=tx.currency,
            transaction_type=tx.transaction_type,
            transaction_created_at=tx.transaction_created_at,
        )
        db.add(event)
        await db.flush()  # get the generated ID without committing

        # ── Score ─────────────────────────────────────────────────────────────
        result = await self.fraud_engine.score(tx, db)

        # ── Persist fraud case if flagged ────────────────────────────────────
        if result.should_flag:
            case = FraudCase(
                transaction_event_id=event.id,
                fraud_score=Decimal(str(result.fraud_score)),
                ml_anomaly_score=Decimal(str(result.ml_anomaly_score)) if result.ml_anomaly_score is not None else None,
                status=CaseStatus.PENDING,
            )
            db.add(case)
            await db.flush()

            for rule in result.fired_rules:
                db.add(FraudRule(
                    fraud_case_id=case.id,
                    rule_name=rule.rule,
                    detail=rule.detail,
                    score_contribution=Decimal(str(rule.score_contribution)),
                ))

            logger.warning(
                "Fraud case created for transaction %s | score=%.4f | rules=%s",
                tx.transaction_id,
                result.fraud_score,
                [r.rule.value for r in result.fired_rules],
            )

        return event
