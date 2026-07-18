"""
Isolation Forest anomaly scorer.

Isolation Forest works by randomly partitioning the feature space.
Anomalous points (rare, extreme values) require fewer splits to isolate,
so they receive a lower (more negative) anomaly score.

Features used:
  - amount          : transaction size
  - hour_of_day     : 0-23 (off-hours is a fraud signal)
  - is_weekend      : 0/1
  - amount_log      : log(amount) — compresses the long tail

The model is trained offline on synthetic data (ml/train.py) and loaded
at startup. If no model file exists, the scorer degrades gracefully —
rule-based checks still run, ML is skipped.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np

from app.schemas.transaction import TransactionEventIn

logger = logging.getLogger(__name__)


class MLScorer:
    def __init__(self, model_path: str = "ml/isolation_forest.pkl"):
        self.model_path = model_path
        self._model = None
        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            try:
                self._model = joblib.load(self.model_path)
                logger.info("Loaded Isolation Forest model from %s", self.model_path)
            except Exception as e:
                logger.warning("Failed to load ML model: %s — scoring will be skipped", e)
        else:
            logger.warning(
                "No ML model found at %s. Run ml/train.py to generate one. "
                "Rule-based scoring will still function normally.",
                self.model_path,
            )

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def score(self, tx: TransactionEventIn) -> float:
        """
        Returns the raw Isolation Forest decision_function score.
        Negative = anomalous, positive = normal.
        """
        if not self.is_ready:
            return 0.0
        features = self._extract_features(tx)
        score = self._model.decision_function([features])[0]
        return float(score)

    def _extract_features(self, tx: TransactionEventIn) -> list:
        amount = float(tx.amount)
        ts = tx.transaction_created_at or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return [
            amount,
            float(np.log1p(amount)),   # log-compress to handle outliers
            float(ts.hour),
            float(1 if ts.weekday() >= 5 else 0),
        ]
