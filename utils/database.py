from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime

# Get database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

class Menu(Base):
    __tablename__ = "menus"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    content = Column(JSON)  # Store the menu data as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

class SubstitutionRule(Base):
    __tablename__ = "substitution_rules"
    
    id = Column(Integer, primary_key=True, index=True)
    allergen = Column(String, index=True)
    original = Column(String)
    replacement = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
def init_db():
    Base.metadata.create_all(bind=engine)

# Database session context manager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
