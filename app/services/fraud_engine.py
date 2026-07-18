"""
Fraud scoring engine.

Two-layer design:
  1. Rule-based checks  — deterministic, fast, human-auditable
  2. ML anomaly score   — Isolation Forest trained on transaction features

Final fraud_score = clamp(rule_score + ml_penalty, 0, 1)
A case is created when fraud_score >= 0.4 OR any hard rule fires.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Tuple

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.fraud_case import RuleName, TransactionEvent
from app.schemas.transaction import TransactionEventIn
from app.services.ml_scorer import MLScorer

logger = logging.getLogger(__name__)
settings = get_settings()

FRAUD_CASE_THRESHOLD = 0.4  # create a case above this score


@dataclass
class RuleResult:
    rule: RuleName
    fired: bool
    detail: str
    score_contribution: float


@dataclass
class ScoringResult:
    fraud_score: float
    ml_anomaly_score: float | None
    rules: List[RuleResult] = field(default_factory=list)

    @property
    def should_flag(self) -> bool:
        return self.fraud_score >= FRAUD_CASE_THRESHOLD or any(
            r.fired and r.score_contribution >= 0.5 for r in self.rules
        )

    @property
    def fired_rules(self) -> List[RuleResult]:
        return [r for r in self.rules if r.fired]


class FraudEngine:
    """
    Stateless scoring engine. Inject a DB session per request so it can
    query historical transaction patterns for the same account.
    """

    def __init__(self, ml_scorer: MLScorer):
        self.ml_scorer = ml_scorer

    async def score(self, tx: TransactionEventIn, db: AsyncSession) -> ScoringResult:
        rules: List[RuleResult] = []

        # ── Rule 1: High-amount threshold ─────────────────────────────────────
        # Transactions >= $10k trigger mandatory SAR review at US banks.
        amount = float(tx.amount)
        rules.append(await self._check_high_amount(amount))

        # ── Rule 2: Round-amount suspicion ────────────────────────────────────
        # Structuring (breaking large amounts into round-number chunks) is a
        # common money-laundering technique. Round amounts are a soft signal.
        rules.append(self._check_round_amount(amount))

        # ── Rule 3: Velocity check ────────────────────────────────────────────
        # More than N transactions from the same account in the last hour
        # suggests account takeover or automated fraud.
        rules.append(await self._check_velocity(tx.source_account_id, db))

        # ── Rule 4: Off-hours transaction ─────────────────────────────────────
        # Fraud is disproportionately common between midnight and 5am.
        rules.append(self._check_off_hours(tx.transaction_created_at))

        # ── ML anomaly score ──────────────────────────────────────────────────
        ml_score = None
        ml_rule = None
        if self.ml_scorer.is_ready:
            ml_score = self.ml_scorer.score(tx)
            ml_rule = self._ml_rule_result(ml_score)
            rules.append(ml_rule)

        # ── Aggregate score ───────────────────────────────────────────────────
        total = sum(r.score_contribution for r in rules if r.fired)
        fraud_score = min(total, 1.0)

        result = ScoringResult(
            fraud_score=round(fraud_score, 4),
            ml_anomaly_score=round(ml_score, 4) if ml_score is not None else None,
            rules=rules,
        )

        if result.should_flag:
            logger.warning(
                "Transaction %s flagged: score=%.4f rules=%s",
                tx.transaction_id,
                fraud_score,
                [r.rule.value for r in result.fired_rules],
            )

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Individual rules
    # ─────────────────────────────────────────────────────────────────────────

    async def _check_high_amount(self, amount: float) -> RuleResult:
        fired = amount >= settings.amount_threshold
        return RuleResult(
            rule=RuleName.HIGH_AMOUNT,
            fired=fired,
            detail=f"Amount ${amount:,.2f} exceeds SAR threshold ${settings.amount_threshold:,.2f}" if fired else "",
            score_contribution=0.6 if fired else 0.0,
        )

    def _check_round_amount(self, amount: float) -> RuleResult:
        """
        Flag transactions that are exactly divisible by the threshold
        (e.g. $1000, $5000, $10000). Structuring often uses clean numbers.
        """
        threshold = settings.round_amount_threshold
        fired = amount >= threshold and (amount % threshold == 0)
        return RuleResult(
            rule=RuleName.ROUND_AMOUNT,
            fired=fired,
            detail=f"Amount ${amount:,.2f} is a round number (possible structuring)" if fired else "",
            score_contribution=0.2 if fired else 0.0,
        )

    async def _check_velocity(self, account_id: str, db: AsyncSession) -> RuleResult:
        """
        Count transactions from this account in the last hour.
        """
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        result = await db.execute(
            select(func.count(TransactionEvent.id))
            .where(TransactionEvent.source_account_id == account_id)
            .where(TransactionEvent.received_at >= one_hour_ago)
        )
        count = result.scalar() or 0
        fired = count >= settings.velocity_max_tx_per_hour
        return RuleResult(
            rule=RuleName.VELOCITY,
            fired=fired,
            detail=f"Account sent {count} transactions in the last hour (max {settings.velocity_max_tx_per_hour})" if fired else "",
            score_contribution=0.5 if fired else 0.0,
        )

    def _check_off_hours(self, created_at: datetime | None) -> RuleResult:
        """
        Transactions between midnight and 5am local UTC are higher risk.
        """
        ts = created_at or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        hour = ts.hour
        fired = 0 <= hour < 5
        return RuleResult(
            rule=RuleName.OFF_HOURS,
            fired=fired,
            detail=f"Transaction initiated at {ts.strftime('%H:%M UTC')} (off-hours window 00:00-05:00)" if fired else "",
            score_contribution=0.2 if fired else 0.0,
        )

    def _ml_rule_result(self, ml_score: float) -> RuleResult:
        """
        Isolation Forest returns negative scores for anomalies.
        Scores below threshold are flagged.
        """
        fired = ml_score < settings.ml_anomaly_score_threshold
        return RuleResult(
            rule=RuleName.ML_ANOMALY,
            fired=fired,
            detail=f"Isolation Forest anomaly score {ml_score:.4f} below threshold {settings.ml_anomaly_score_threshold}" if fired else "",
            score_contribution=0.4 if fired else 0.0,
        )
