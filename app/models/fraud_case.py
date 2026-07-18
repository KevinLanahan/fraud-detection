from sqlalchemy import Column, String, Numeric, DateTime, Integer, Boolean, Text, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base
import uuid
import enum
from datetime import datetime, timezone


class CaseStatus(str, enum.Enum):
    PENDING = "PENDING"
    REVIEWED = "REVIEWED"
    CONFIRMED_FRAUD = "CONFIRMED_FRAUD"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class RuleName(str, enum.Enum):
    VELOCITY = "VELOCITY"
    HIGH_AMOUNT = "HIGH_AMOUNT"
    ROUND_AMOUNT = "ROUND_AMOUNT"
    OFF_HOURS = "OFF_HOURS"
    ML_ANOMALY = "ML_ANOMALY"


class TransactionEvent(Base):
    """
    Mirror of a transaction received from Project 1 via Redis Pub/Sub.
    Stored so the fraud engine can query historical patterns per account.
    """
    __tablename__ = "transaction_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(String(36), nullable=False, unique=True, index=True)
    source_account_id = Column(String(36), nullable=False, index=True)
    dest_account_id = Column(String(36), nullable=False, index=True)
    amount = Column(Numeric(19, 4), nullable=False)
    currency = Column(String(3), nullable=False)
    transaction_type = Column(String(30), nullable=False)
    received_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    transaction_created_at = Column(DateTime(timezone=True), nullable=True)

    fraud_case = relationship("FraudCase", back_populates="transaction_event", uselist=False)


class FraudCase(Base):
    """
    A flagged transaction under review.
    Created whenever fraud_score > threshold or any hard rule fires.
    """
    __tablename__ = "fraud_cases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_event_id = Column(UUID(as_uuid=True), ForeignKey("transaction_events.id"), nullable=False)
    fraud_score = Column(Numeric(5, 4), nullable=False)   # 0.0 – 1.0
    ml_anomaly_score = Column(Numeric(8, 4), nullable=True)  # raw IsolationForest score
    status = Column(SAEnum(CaseStatus), nullable=False, default=CaseStatus.PENDING)
    reviewer_notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    reviewed_at = Column(DateTime(timezone=True), nullable=True)

    transaction_event = relationship("TransactionEvent", back_populates="fraud_case")
    rules_fired = relationship("FraudRule", back_populates="fraud_case", cascade="all, delete-orphan")


class FraudRule(Base):
    """
    Which specific rules fired for a given fraud case.
    Gives reviewers an audit trail of exactly why a transaction was flagged.
    """
    __tablename__ = "fraud_rules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fraud_case_id = Column(UUID(as_uuid=True), ForeignKey("fraud_cases.id"), nullable=False)
    rule_name = Column(SAEnum(RuleName), nullable=False)
    detail = Column(Text, nullable=True)   # human-readable explanation
    score_contribution = Column(Numeric(5, 4), nullable=False)

    fraud_case = relationship("FraudCase", back_populates="rules_fired")
