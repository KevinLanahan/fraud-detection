from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from datetime import datetime, timezone
from typing import List, Optional
import uuid

from app.database import get_db
from app.models.fraud_case import FraudCase, TransactionEvent, CaseStatus
from app.schemas.fraud_case import FraudCaseOut, ReviewRequest

router = APIRouter(prefix="/api/v1/cases", tags=["fraud-cases"])


@router.get("", response_model=List[FraudCaseOut])
async def list_cases(
    status: Optional[CaseStatus] = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List fraud cases, optionally filtered by status."""
    q = (
        select(FraudCase)
        .options(
            selectinload(FraudCase.transaction_event),
            selectinload(FraudCase.rules_fired),
        )
        .order_by(FraudCase.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if status:
        q = q.where(FraudCase.status == status)

    result = await db.execute(q)
    cases = result.scalars().all()
    return [_to_out(c) for c in cases]


@router.get("/{case_id}", response_model=FraudCaseOut)
async def get_case(case_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(FraudCase)
        .options(
            selectinload(FraudCase.transaction_event),
            selectinload(FraudCase.rules_fired),
        )
        .where(FraudCase.id == uuid.UUID(case_id))
    )
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Fraud case not found")
    return _to_out(case)


@router.patch("/{case_id}/review", response_model=FraudCaseOut)
async def review_case(
    case_id: str,
    body: ReviewRequest,
    db: AsyncSession = Depends(get_db),
):
    """Mark a fraud case as reviewed — confirm fraud or clear as false positive."""
    result = await db.execute(
        select(FraudCase)
        .options(
            selectinload(FraudCase.transaction_event),
            selectinload(FraudCase.rules_fired),
        )
        .where(FraudCase.id == uuid.UUID(case_id))
    )
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Fraud case not found")

    case.status = body.status
    case.reviewer_notes = body.reviewer_notes
    case.reviewed_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(case)
    return _to_out(case)


def _to_out(case: FraudCase) -> FraudCaseOut:
    tx = case.transaction_event
    return FraudCaseOut(
        id=str(case.id),
        transaction_id=tx.transaction_id,
        source_account_id=tx.source_account_id,
        dest_account_id=tx.dest_account_id,
        amount=tx.amount,
        currency=tx.currency,
        fraud_score=case.fraud_score,
        ml_anomaly_score=case.ml_anomaly_score,
        status=case.status,
        rules_fired=case.rules_fired,
        created_at=case.created_at,
        reviewed_at=case.reviewed_at,
        reviewer_notes=case.reviewer_notes,
    )
