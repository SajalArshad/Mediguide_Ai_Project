"""
cache_manager.py
Demonstrates BOTH InMemoryCache and SQLiteCache for LangChain LLM calls.

InMemoryCache: lives in RAM, fastest, cleared on app restart -- best for a
single running session.

SQLiteCache: persisted to a .db file on disk, slightly slower than memory
but survives restarts -- best for reusing identical results across sessions.

Register a cache once with set_llm_cache(...); LangChain then checks it
automatically before every model call, so submitting the same form twice
should be visibly faster the second time.
"""

import os
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache

CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), "..", ".mediguide_cache.db")


def enable_in_memory_cache() -> None:
    set_llm_cache(InMemoryCache())


def enable_sqlite_cache(db_path: str = CACHE_DB_PATH) -> None:
    set_llm_cache(SQLiteCache(database_path=db_path))


def disable_cache() -> None:
    set_llm_cache(None)


def apply_cache_choice(choice: str) -> None:
    """choice is one of: 'No caching', 'In-memory cache', 'SQLite cache'."""
    if choice == "In-memory cache":
        enable_in_memory_cache()
    elif choice == "SQLite cache":
        enable_sqlite_cache()
    else:
        disable_cache()
