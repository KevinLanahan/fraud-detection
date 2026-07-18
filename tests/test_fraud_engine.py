"""
Tests for the rule-based fraud engine.

All tests use an in-memory SQLite database with a mocked async session,
so no running Postgres or Redis is required.
"""
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.fraud_engine import FraudEngine, FRAUD_CASE_THRESHOLD
from app.services.ml_scorer import MLScorer
from app.schemas.transaction import TransactionEventIn
from app.models.fraud_case import RuleName


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tx(
    amount: float = 500.0,
    hour: int = 14,
    source: str = "acc-111",
    dest: str = "acc-222",
) -> TransactionEventIn:
    ts = datetime(2024, 6, 1, hour, 0, 0, tzinfo=timezone.utc)
    return TransactionEventIn(
        transaction_id="tx-abc-123",
        source_account_id=source,
        dest_account_id=dest,
        amount=Decimal(str(amount)),
        currency="USD",
        transaction_type="TRANSFER",
        transaction_created_at=ts,
    )


def _engine(ml_ready: bool = False, ml_score: float = 0.0) -> FraudEngine:
    scorer = MagicMock(spec=MLScorer)
    scorer.is_ready = ml_ready
    scorer.score.return_value = ml_score
    return FraudEngine(scorer)


def _mock_db(velocity_count: int = 0) -> AsyncMock:
    """Async session that returns `velocity_count` for COUNT queries.

    SQLAlchemy's CursorResult.scalar() is synchronous after an awaited execute,
    so we use MagicMock (not AsyncMock) for it.
    """
    execute_result = MagicMock()
    execute_result.scalar = MagicMock(return_value=velocity_count)
    db = AsyncMock()
    db.execute = AsyncMock(return_value=execute_result)
    return db


# ── HIGH_AMOUNT rule ──────────────────────────────────────────────────────────

class TestHighAmountRule:
    @pytest.mark.asyncio
    async def test_fires_above_threshold(self):
        engine = _engine()
        result = await engine.score(_tx(amount=15_000.0), _mock_db())
        high_rule = next(r for r in result.rules if r.rule == RuleName.HIGH_AMOUNT)
        assert high_rule.fired is True
        assert high_rule.score_contribution == 0.6

    @pytest.mark.asyncio
    async def test_does_not_fire_below_threshold(self):
        engine = _engine()
        result = await engine.score(_tx(amount=500.0), _mock_db())
        high_rule = next(r for r in result.rules if r.rule == RuleName.HIGH_AMOUNT)
        assert high_rule.fired is False
        assert high_rule.score_contribution == 0.0

    @pytest.mark.asyncio
    async def test_fires_at_exact_threshold(self):
        engine = _engine()
        result = await engine.score(_tx(amount=10_000.0), _mock_db())
        high_rule = next(r for r in result.rules if r.rule == RuleName.HIGH_AMOUNT)
        assert high_rule.fired is True


# ── ROUND_AMOUNT rule ─────────────────────────────────────────────────────────

class TestRoundAmountRule:
    @pytest.mark.asyncio
    async def test_fires_on_round_large_amount(self):
        engine = _engine()
        result = await engine.score(_tx(amount=5_000.0), _mock_db())
        rule = next(r for r in result.rules if r.rule == RuleName.ROUND_AMOUNT)
        assert rule.fired is True

    @pytest.mark.asyncio
    async def test_does_not_fire_on_small_round_amount(self):
        engine = _engine()
        result = await engine.score(_tx(amount=100.0), _mock_db())
        rule = next(r for r in result.rules if r.rule == RuleName.ROUND_AMOUNT)
        assert rule.fired is False

    @pytest.mark.asyncio
    async def test_does_not_fire_on_unround_amount(self):
        engine = _engine()
        result = await engine.score(_tx(amount=1_234.56), _mock_db())
        rule = next(r for r in result.rules if r.rule == RuleName.ROUND_AMOUNT)
        assert rule.fired is False


# ── VELOCITY rule ─────────────────────────────────────────────────────────────

class TestVelocityRule:
    @pytest.mark.asyncio
    async def test_fires_when_velocity_exceeded(self):
        engine = _engine()
        result = await engine.score(_tx(), _mock_db(velocity_count=15))
        rule = next(r for r in result.rules if r.rule == RuleName.VELOCITY)
        assert rule.fired is True
        assert rule.score_contribution == 0.5

    @pytest.mark.asyncio
    async def test_does_not_fire_when_below_velocity(self):
        engine = _engine()
        result = await engine.score(_tx(), _mock_db(velocity_count=3))
        rule = next(r for r in result.rules if r.rule == RuleName.VELOCITY)
        assert rule.fired is False

    @pytest.mark.asyncio
    async def test_velocity_flags_case(self):
        engine = _engine()
        result = await engine.score(_tx(), _mock_db(velocity_count=20))
        assert result.should_flag is True


# ── OFF_HOURS rule ────────────────────────────────────────────────────────────

class TestOffHoursRule:
    @pytest.mark.asyncio
    async def test_fires_at_midnight(self):
        engine = _engine()
        result = await engine.score(_tx(hour=0), _mock_db())
        rule = next(r for r in result.rules if r.rule == RuleName.OFF_HOURS)
        assert rule.fired is True

    @pytest.mark.asyncio
    async def test_fires_at_4am(self):
        engine = _engine()
        result = await engine.score(_tx(hour=4), _mock_db())
        rule = next(r for r in result.rules if r.rule == RuleName.OFF_HOURS)
        assert rule.fired is True

    @pytest.mark.asyncio
    async def test_does_not_fire_at_5am(self):
        engine = _engine()
        result = await engine.score(_tx(hour=5), _mock_db())
        rule = next(r for r in result.rules if r.rule == RuleName.OFF_HOURS)
        assert rule.fired is False

    @pytest.mark.asyncio
    async def test_does_not_fire_during_business_hours(self):
        engine = _engine()
        result = await engine.score(_tx(hour=14), _mock_db())
        rule = next(r for r in result.rules if r.rule == RuleName.OFF_HOURS)
        assert rule.fired is False


# ── ML anomaly rule ───────────────────────────────────────────────────────────

class TestMLRule:
    @pytest.mark.asyncio
    async def test_skipped_when_model_not_ready(self):
        engine = _engine(ml_ready=False)
        result = await engine.score(_tx(), _mock_db())
        rule_names = [r.rule for r in result.rules]
        assert RuleName.ML_ANOMALY not in rule_names

    @pytest.mark.asyncio
    async def test_fires_when_anomalous_score(self):
        engine = _engine(ml_ready=True, ml_score=-0.15)
        result = await engine.score(_tx(), _mock_db())
        rule = next(r for r in result.rules if r.rule == RuleName.ML_ANOMALY)
        assert rule.fired is True
        assert rule.score_contribution == 0.4

    @pytest.mark.asyncio
    async def test_does_not_fire_on_normal_score(self):
        engine = _engine(ml_ready=True, ml_score=0.05)
        result = await engine.score(_tx(), _mock_db())
        rule = next(r for r in result.rules if r.rule == RuleName.ML_ANOMALY)
        assert rule.fired is False


# ── Aggregate scoring ─────────────────────────────────────────────────────────

class TestAggregate:
    @pytest.mark.asyncio
    async def test_clean_transaction_not_flagged(self):
        """Normal transaction: $500 daytime, low velocity — should not be flagged."""
        engine = _engine()
        result = await engine.score(_tx(amount=500.0, hour=14), _mock_db(velocity_count=2))
        assert result.should_flag is False
        assert result.fraud_score < FRAUD_CASE_THRESHOLD

    @pytest.mark.asyncio
    async def test_score_caps_at_1(self):
        """Multiple rules all firing should not exceed fraud_score of 1.0."""
        engine = _engine(ml_ready=True, ml_score=-0.5)
        result = await engine.score(
            _tx(amount=15_000.0, hour=2),
            _mock_db(velocity_count=20),
        )
        assert result.fraud_score <= 1.0

    @pytest.mark.asyncio
    async def test_high_amount_alone_flags_case(self):
        """$15k alone should be enough to flag (score 0.6 >= threshold 0.4)."""
        engine = _engine()
        result = await engine.score(_tx(amount=15_000.0, hour=14), _mock_db(velocity_count=1))
        assert result.should_flag is True

    @pytest.mark.asyncio
    async def test_fired_rules_list(self):
        """fired_rules should only include rules that actually fired."""
        engine = _engine()
        result = await engine.score(_tx(amount=15_000.0, hour=2), _mock_db(velocity_count=1))
        fired_names = {r.rule for r in result.fired_rules}
        assert RuleName.HIGH_AMOUNT in fired_names
        assert RuleName.OFF_HOURS in fired_names
        # VELOCITY should not fire (count=1)
        assert RuleName.VELOCITY not in fired_names


# ── MLScorer unit tests ───────────────────────────────────────────────────────

class TestMLScorer:
    def test_not_ready_when_no_model(self, tmp_path):
        scorer = MLScorer(str(tmp_path / "nonexistent.pkl"))
        assert scorer.is_ready is False

    def test_score_returns_zero_when_not_ready(self, tmp_path):
        scorer = MLScorer(str(tmp_path / "nonexistent.pkl"))
        result = scorer.score(_tx())
        assert result == 0.0

    def test_loads_model_from_file(self, tmp_path):
        import joblib
        from sklearn.ensemble import IsolationForest
        import numpy as np

        model = IsolationForest(n_estimators=10, random_state=42)
        model.fit(np.random.randn(50, 4))
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)

        scorer = MLScorer(str(model_path))
        assert scorer.is_ready is True
        score = scorer.score(_tx(amount=500.0, hour=14))
        assert isinstance(score, float)
