"""Conexión a SQL Server (AdventureWorks2022) usando pymssql + SQLAlchemy."""
import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def get_engine():
    user = os.getenv("DB_USER")
    pwd = quote_plus(os.getenv("DB_PASSWORD", ""))
    server = os.getenv("DB_SERVER")
    port = os.getenv("DB_PORT", "1433")
    db = os.getenv("DB_NAME", "AdventureWorks2022")
    return create_engine(f"mssql+pymssql://{user}:{pwd}@{server}:{port}/{db}")


def read_sql(query: str) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn)
