from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# We'll get Base from database.py to avoid circular imports if models grow
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
