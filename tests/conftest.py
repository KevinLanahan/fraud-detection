"""
Shared pytest configuration for the fraud-detection test suite.

Tests are designed to run without a live database or Redis:
  - Unit tests use mocked async sessions.
  - Integration-style tests (if added later) can use pytest-asyncio's
    event_loop fixture and an in-memory SQLite engine.
"""
import os

# Point to a dummy model path so MLScorer doesn't log noise during tests
os.environ.setdefault("ML_MODEL_PATH", "/tmp/nonexistent_test_model.pkl")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6380")
