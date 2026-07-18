from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database import get_db
from app.models.fraud_case import FraudCase, TransactionEvent, CaseStatus
from app.schemas.fraud_case import DashboardStats

router = APIRouter(tags=["dashboard"])


@router.get("/api/v1/stats", response_model=DashboardStats)
async def get_stats(db: AsyncSession = Depends(get_db)):
    total_tx = (await db.execute(select(func.count(TransactionEvent.id)))).scalar() or 0
    total_flagged = (await db.execute(select(func.count(FraudCase.id)))).scalar() or 0
    pending = (await db.execute(select(func.count(FraudCase.id)).where(FraudCase.status == CaseStatus.PENDING))).scalar() or 0
    confirmed = (await db.execute(select(func.count(FraudCase.id)).where(FraudCase.status == CaseStatus.CONFIRMED_FRAUD))).scalar() or 0
    fp = (await db.execute(select(func.count(FraudCase.id)).where(FraudCase.status == CaseStatus.FALSE_POSITIVE))).scalar() or 0

    return DashboardStats(
        total_transactions_seen=total_tx,
        total_flagged=total_flagged,
        pending_review=pending,
        confirmed_fraud=confirmed,
        false_positives=fp,
        flag_rate_pct=round((total_flagged / total_tx * 100) if total_tx else 0, 2),
    )


@router.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("static/dashboard.html") as f:
        return f.read()
