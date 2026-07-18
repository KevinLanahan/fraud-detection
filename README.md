# Fraud Detection Pipeline

A real-time fraud detection service that consumes payment transaction events from a Redis Pub/Sub channel, scores them using a rule engine combined with an Isolation Forest ML model, and surfaces flagged cases through a fraud analyst review dashboard.

Designed as a downstream consumer of the [payment-processor](https://github.com/KevinLanahan/payment-processor) service.

## Architecture

```
payment-processor (Project 1)
    │
    └── Redis Pub/Sub (transactions:completed)
            │
            ▼
    Redis Consumer (async, exponential backoff reconnect)
            │
            ▼
    Ingest Service
    ├── Idempotency check        Skip already-processed transactions
    ├── Rule Engine              5 weighted rules → fraud score
    │   ├── HIGH_AMOUNT          >= $10k (SAR threshold) → +0.6
    │   ├── VELOCITY             >= 10 tx/hr from same account → +0.5
    │   ├── ROUND_AMOUNT         Round numbers >= $1k (structuring) → +0.2
    │   ├── OFF_HOURS            Midnight–5am UTC → +0.2
    │   └── ML_ANOMALY           Isolation Forest score below threshold → +0.4
    ├── Isolation Forest         Trained on 5,250 synthetic transactions
    └── FraudCase persistence    Created when score >= 0.4 or hard rule fires
            │
            ▼
    Review Dashboard (http://localhost:8081)
```

## Key Engineering Decisions

**Two-layer scoring** — Rule-based checks provide deterministic, auditable signals (HIGH_AMOUNT maps directly to US Bank Secrecy Act SAR requirements). The ML layer catches anomalies that don't match explicit rules. Combining both reduces false negatives while keeping decisions explainable.

**Isolation Forest** — Unsupervised anomaly detection ideal for fraud where labeled fraud examples are scarce. Trained on features: transaction amount, log(amount), hour of day, is_weekend. Points requiring fewer splits to isolate are scored as anomalous.

**Idempotent ingestion** — Transactions are deduplicated by `transaction_id` before scoring. The Redis consumer can reconnect and replay messages without creating duplicate fraud cases.

**Async throughout** — FastAPI + async SQLAlchemy + asyncpg keeps the event loop unblocked. The Redis consumer runs as a background asyncio task with exponential backoff (2s → 60s cap) on connection failures.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Framework | FastAPI |
| Database | PostgreSQL 16 + async SQLAlchemy |
| ML | scikit-learn (Isolation Forest) |
| Events | Redis Pub/Sub |
| Testing | pytest, pytest-asyncio |
| Infrastructure | Docker, Docker Compose |

## Getting Started

**Prerequisites:** Docker, Docker Compose

```bash
git clone https://github.com/KevinLanahan/fraud-detection
cd fraud-detection
docker compose up -d

# Train the ML model (required on first run)
docker exec fraud-detection-app python3 ml/train.py

# Dashboard
open http://localhost:8081
```

## API

### Ingest a transaction manually
```bash
curl -X POST http://localhost:8081/api/v1/ingest/transaction \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "test-001",
    "source_account_id": "acc-aaa",
    "dest_account_id": "acc-bbb",
    "amount": "15000.00",
    "currency": "USD",
    "transaction_type": "TRANSFER",
    "transaction_created_at": "2024-06-01T02:30:00Z"
  }'
```

### List fraud cases
```bash
curl "http://localhost:8081/api/v1/cases?status=PENDING"
```

### Review a case
```bash
curl -X PATCH http://localhost:8081/api/v1/cases/<case-id>/review \
  -H "Content-Type: application/json" \
  -d '{"status": "CONFIRMED_FRAUD", "reviewer_notes": "Verified SAR threshold breach"}'
```

### Dashboard stats
```bash
curl http://localhost:8081/api/v1/stats
```

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

23 tests covering all 5 fraud rules, ML scorer loading, score aggregation, and edge cases.

## Fraud Rules

| Rule | Condition | Score Contribution |
|---|---|---|
| HIGH_AMOUNT | Amount >= $10,000 | +0.60 |
| VELOCITY | >= 10 transactions/hour from same account | +0.50 |
| ROUND_AMOUNT | Round number >= $1,000 | +0.20 |
| OFF_HOURS | Transaction time 00:00–05:00 UTC | +0.20 |
| ML_ANOMALY | Isolation Forest score below threshold | +0.40 |

A fraud case is created when total score >= 0.4 or any single hard rule scores >= 0.5.

## Related Projects

- [payment-processor](https://github.com/KevinLanahan/payment-processor) — publishes transaction events consumed by this service
- [portfolio-risk](https://github.com/KevinLanahan/portfolio-risk) — standalone portfolio risk calculator
