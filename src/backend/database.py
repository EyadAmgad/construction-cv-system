"""
Database connection pool for the FastAPI backend.
Uses asyncpg for async PostgreSQL access.
"""

import os
import asyncpg

DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 5432))
DB_NAME     = os.getenv("DB_NAME",     "construction_cv")
DB_USER     = os.getenv("DB_USER",     "cvuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "cvpass")

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            min_size=2,
            max_size=10,
        )
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
