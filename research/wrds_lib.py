"""WRDS connection helper. Reads WRDS_USERNAME / WRDS_PASSWORD from .env (same
secure pattern as the Databento key) and connects to the WRDS cloud PostgreSQL
over SSL. Provides q() for pandas queries.

Add to .env:
    WRDS_USERNAME=your_wrds_user
    WRDS_PASSWORD=your_wrds_password
"""
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from options_scout.envfile import load_env  # noqa: E402

HOST, PORT, DB = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"
_engine = None


def engine():
    global _engine
    if _engine is None:
        load_env()
        u = os.environ.get("WRDS_USERNAME")
        p = os.environ.get("WRDS_PASSWORD")
        if not u or not p:
            raise SystemExit("Set WRDS_USERNAME and WRDS_PASSWORD in .env first.")
        url = f"postgresql+psycopg2://{u}:{p}@{HOST}:{PORT}/{DB}?sslmode=require"
        _engine = create_engine(url, connect_args={"connect_timeout": 30})
    return _engine


def q(sql, **params):
    """Run a SQL query, return a DataFrame."""
    with engine().connect() as c:
        return pd.read_sql(text(sql), c, params=params)
