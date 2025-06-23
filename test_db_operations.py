# test_db_operations.py

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

# Import Base and models from your existing files
from db import Base  # Assuming database.py provides the Base
from models import Conversation, Message, Citation, Reaction

# Import the database operations functions
import db_operations


@pytest.fixture(scope="function")
def db_session():
    """
    Provides a SQLAlchemy session for testing.
    Each test will get a fresh in-memory SQLite database.
    """
    # Use an in-memory SQLite database for testing
    TEST_DATABASE_URL = "sqlite:///:memory:"
    engine = create_engine(TEST_DATABASE_URL, echo=False)  # echo=True for debugging SQL

    # Create all tables for the test database
    db_operations.initialize_db(engine)

    # Create a session local for this engine
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()

    yield session  # Provide the session to the test

    # Teardown: Close the session and drop all tables after the test
    session.close()
    db_operations.delete_all_tables(engine)


def test_db_initialization(db_session):
    """
    Tests if the database tables are created correctly.
    """
    inspector = inspect(db_session.bind)
    table_names = inspector.get_table_names()

    # Check if all expected tables exist
    assert "conversation" in table_names
    assert "message" in table_names
    assert "citation" in table_names
    assert "reactions" in table_names  # Your DBML uses 'reactions'

    # Check if there are no unexpected tables (useful in real scenarios)
    assert len(table_names) == 4


def test_fill_with_sample_data(db_session):
    """
    Tests if fill_with_sample_data correctly populates the database.
    """
    db_operations.fill_with_sample_data(db_session)

    # Verify counts
    assert len(db_operations.get_all_conversations(db_session)) == 1
    assert len(db_operations.get_all_messages(db_session)) == 2
    assert len(db_operations.get_all_citations(db_session)) == 2
    assert len(db_operations.get_all_reactions(db_session)) == 2

    # Verify content of one record
    conversation = db_operations.get_all_conversations(db_session)[0]
    assert conversation.conversation_name == "AI Bot Debate on Space Exploration"
    assert len(conversation.messages) == 2  # Check relationship loading

    message = db_operations.get_all_messages(db_session)[0]
    assert message.writer == "AstroBot"
    assert len(message.citations) == 2  # Check relationship loading


def test_clear_all_table_data(db_session):
    """
    Tests if clear_all_table_data correctly clears all data.
    """
    db_operations.fill_with_sample_data(db_session)  # First, add some data
    assert len(db_operations.get_all_conversations(db_session)) > 0  # Ensure data exists

    db_operations.clear_all_table_data(db_session)

    # Verify all tables are empty
    assert len(db_operations.get_all_conversations(db_session)) == 0
    assert len(db_operations.get_all_messages(db_session)) == 0
    assert len(db_operations.get_all_citations(db_session)) == 0
    assert len(db_operations.get_all_reactions(db_session)) == 0

    # Verify tables still exist (structure not deleted)
    inspector = inspect(db_session.bind)
    table_names = inspector.get_table_names()
    assert "conversation" in table_names
    assert "message" in table_names


def test_delete_all_tables(db_session):
    """
    Tests if delete_all_tables correctly drops all tables.
    """
    # Tables are created by the fixture, so they exist initially
    inspector = inspect(db_session.bind)
    assert "conversation" in inspector.get_table_names()

    db_operations.delete_all_tables(db_session.bind)  # Pass the engine

    # Verify all tables are gone
    inspector = inspect(db_session.bind)  # Re-inspect after deletion
    assert len(inspector.get_table_names()) == 0


def test_get_all_queries(db_session):
    """
    Tests the basic get_all_ functions.
    """
    db_operations.fill_with_sample_data(db_session)

    conversations = db_operations.get_all_conversations(db_session)
    messages = db_operations.get_all_messages(db_session)
    citations = db_operations.get_all_citations(db_session)
    reactions = db_operations.get_all_reactions(db_session)

    assert len(conversations) == 1
    assert len(messages) == 2
    assert len(citations) == 2
    assert len(reactions) == 2

    # Check some basic attributes of retrieved objects
    assert conversations[0].bot_1_name == "AstroBot"
    assert messages[0].topic == "Mars Colonization"
    assert citations[0].chunk.startswith("NASA's")
    assert reactions[0].reaction_name == "💚"
