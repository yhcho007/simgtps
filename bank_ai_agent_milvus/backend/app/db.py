"""Simple SQLite-backed storage for interactions and feedback.
- Uses SQLAlchemy ORM for portability.
- In production, replace with managed RDS/Cloud SQL and proper migrations.
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime, os

DB_PATH = os.path.join(os.getcwd(), 'data', 'metrics.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
ENGINE = create_engine(f'sqlite:///{DB_PATH}', connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)
Base = declarative_base()

class Interaction(Base):
    __tablename__ = 'interactions'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    session_id = Column(String, index=True)
    user = Column(String, index=True)
    query = Column(Text)
    reply = Column(Text)
    response_time_ms = Column(Float)
    success = Column(Integer)  # 1 or 0
    satisfaction = Column(Integer, nullable=True)  # 1-10
    feedback = Column(Text, nullable=True)

# Helper functions
def init_db():
    Base.metadata.create_all(bind=ENGINE)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
