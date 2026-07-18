from pydantic import BaseModel, Field
from decimal import Decimal
from datetime import datetime
from typing import Optional


class TransactionEventIn(BaseModel):
    """
    Payload published by Project 1 (payment-processor) to Redis
    when a transaction completes. Also accepted directly via POST /ingest.
    """
    transaction_id: str
    source_account_id: str
    dest_account_id: str
    amount: Decimal = Field(gt=0)
    currency: str
    transaction_type: str
    transaction_created_at: Optional[datetime] = None


class TransactionEventOut(BaseModel):
    id: str
    transaction_id: str
    source_account_id: str
    dest_account_id: str
    amount: Decimal
    currency: str
    transaction_type: str
    received_at: datetime

    model_config = {"from_attributes": True}
