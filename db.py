import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

if os.path.exists("env_config.py"):
    import env_config

if os.path.exists("env_config.py"):
    DATABASE_URL = env_config.DATABASE_URL
else:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data.db")

engine = create_engine(DATABASE_URL, echo=False)

Session = sessionmaker(bind=engine)
Base = declarative_base()


class Conversation(Base):
    __tablename__ = 'conversation'

    id = Column(Integer, primary_key=True)
    conversation_name = Column(String)
    bot_1_name = Column(String)
    bot_1_persona = Column(String)
    bot_1_system = Column(String)
    bot_2_name = Column(String)
    bot_2_persona = Column(String)
    bot_2_system = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    messages = relationship("Message", back_populates="conversation")

    def __repr__(self):
        return f"<Conversation(id={self.id}, name='{self.conversation_name}')>"


class Message(Base):
    __tablename__ = 'message'

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversation.id'))
    message = Column(String)
    upvotes = Column(Integer, default=0)
    writer = Column(String)
    topic = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    conversation = relationship("Conversation", back_populates="messages")
    citations = relationship("Citation", back_populates="message")
    reactions = relationship("Reaction", back_populates="message")

    def __repr__(self):
        return f"<Message(id={self.id}, conversation_id={self.conversation_id}, writer='{self.writer}')>"


class Citation(Base):
    __tablename__ = 'citation'

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey('message.id'))
    chunk = Column(String)

    message = relationship("Message", back_populates="citations")

    def __repr__(self):
        return f"<Citation(id={self.id}, message_id={self.message_id})>"


class Reaction(Base):
    __tablename__ = 'reactions'

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey('message.id'))
    reaction_name = Column(String)
    quantity = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)

    message = relationship("Message", back_populates="reactions")

    def __repr__(self):
        return f"<Reaction(id={self.id}, reaction_name='{self.reaction_name}', quantity={self.quantity})>"

def get_db():
    """
    Dependency function to get a database session.
    It yields a session and ensures it's closed after the request.
    """
    db = Session()
    try:
        yield db
    finally:
        db.close()


def initialize_db():
    """
    Initializes the database by creating all tables defined in Base's metadata.
    This function uses the global 'engine'.
    """
    print(f"Attempting to create database tables on: {engine.url}")
    Base.metadata.create_all(bind=engine)
    print("Database tables created or already exist.")
