from sqlalchemy.orm.session import Session


from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from okc_core.config import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker[Session](bind=engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

@event.listens_for(engine, "connect")
def _enable_extensions(dbapi_connection, connection_record):
    # ensure pgvector + pg_trgm
    with dbapi_connection.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
