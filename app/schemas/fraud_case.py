from pydantic import BaseModel
from decimal import Decimal
from datetime import datetime
from typing import Optional, List
from app.models.fraud_case import CaseStatus, RuleName


class RuleFiredOut(BaseModel):
    rule_name: RuleName
    detail: Optional[str]
    score_contribution: Decimal

    model_config = {"from_attributes": True}


class FraudCaseOut(BaseModel):
    id: str
    transaction_id: str
    source_account_id: str
    dest_account_id: str
    amount: Decimal
    currency: str
    fraud_score: Decimal
    ml_anomaly_score: Optional[Decimal]
    status: CaseStatus
    rules_fired: List[RuleFiredOut]
    created_at: datetime
    reviewed_at: Optional[datetime]
    reviewer_notes: Optional[str]

    model_config = {"from_attributes": True}


class ReviewRequest(BaseModel):
    status: CaseStatus
    reviewer_notes: Optional[str] = None


class DashboardStats(BaseModel):
    total_transactions_seen: int
    total_flagged: int
    pending_review: int
    confirmed_fraud: int
    false_positives: int
    flag_rate_pct: float
