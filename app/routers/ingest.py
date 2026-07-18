from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.transaction import TransactionEventIn, TransactionEventOut
from app.services.ingest_service import IngestService
from app.dependencies import get_ingest_service

router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])


@router.post("/transaction", response_model=dict, status_code=202)
async def ingest_transaction(
    tx: TransactionEventIn,
    db: AsyncSession = Depends(get_db),
    ingest_service: IngestService = Depends(get_ingest_service),
):
    """
    Manually ingest a transaction event for fraud scoring.
    Normally called automatically via the Redis Pub/Sub consumer,
    but also available for direct testing or manual submission.
    """
    event = await ingest_service.process(tx, db)
    await db.commit()
    if event is None:
        return {"status": "duplicate", "message": "Transaction already processed"}
    return {"status": "accepted", "event_id": str(event.id)}
