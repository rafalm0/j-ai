from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Define the base for our declarative models.
# This Base will be imported by models.py
Base = declarative_base()

# SQLite in-memory database for demonstration
DATABASE_URL = "sqlite:///./test.db"  # Use "sqlite:///:memory:" for an in-memory database

engine = create_engine(DATABASE_URL, echo=True)  # echo=True for logging SQL statements

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initializes the database by creating all tables defined in Base's metadata.
    """
    # Import all models to ensure they are registered with Base's metadata
    from .models import Conversation, Message, Citation, Reaction
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")
