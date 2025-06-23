# db_operations.py

from datetime import datetime
from sqlalchemy import inspect
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import Base and models from your existing files
from db import Base  # Assuming database.py provides the Base
from models import Conversation, Message, Citation, Reaction


def initialize_db(engine):
    """
    Initializes the database by creating all tables defined in Base's metadata.
    This function expects an SQLAlchemy engine.
    """
    print(f"Initializing database using engine: {engine.url}")
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")


def delete_all_tables(engine):
    """
    Deletes (drops) all tables from the database. Use with caution!
    This function expects an SQLAlchemy engine.
    """
    print(f"Deleting all tables using engine: {engine.url}")
    Base.metadata.drop_all(bind=engine)
    print("All database tables dropped.")


def clear_all_table_data(session: Session):
    """
    Deletes all data from all tables, but keeps the table structure.
    This function expects an SQLAlchemy session.
    """
    print("Clearing all data from tables...")
    # Order of deletion is important for foreign key constraints
    # Delete from child tables first
    session.query(Citation).delete()
    session.query(Reaction).delete()
    session.query(Message).delete()
    session.query(Conversation).delete()

    session.commit()
    print("All data cleared.")


def fill_with_sample_data(session: Session):
    """
    Fills the database with sample data.
    This function expects an SQLAlchemy session.
    """
    print("Filling database with sample data...")

    # Ensure tables are empty before adding sample data
    clear_all_table_data(session)

    # Create a conversation
    conversation = Conversation(
        conversation_name="AI Bot Debate on Space Exploration",
        bot_1_name="AstroBot",
        bot_1_persona="An enthusiastic proponent of space colonization",
        bot_1_system="Focus on human expansion and resource acquisition.",
        bot_2_name="TerraBot",
        bot_2_persona="A pragmatic advocate for Earth-focused solutions",
        bot_2_system="Emphasize sustainability and immediate global challenges.",
        created_at=datetime.now()
    )
    session.add(conversation)
    session.commit()  # Commit to get conversation.id
    session.refresh(conversation)

    # Add messages to the conversation
    message_astro = Message(
        conversation_id=conversation.id,
        message="Mars colonization is vital for humanity's long-term survival!",
        upvotes=8,
        writer="AstroBot",
        topic="Mars Colonization",
        created_at=datetime.now()
    )
    message_terra = Message(
        conversation_id=conversation.id,
        message="Shouldn't we solve climate change on Earth first?",
        upvotes=4,
        writer="TerraBot",
        topic="Earth Challenges",
        created_at=datetime.now()
    )
    session.add_all([message_astro, message_terra])
    session.commit()  # Commit to get message IDs
    session.refresh(message_astro)
    session.refresh(message_terra)

    # Add citations to AstroBot's message
    citation1 = Citation(
        message_id=message_astro.id,
        chunk="NASA's 'Journey to Mars' initiative outlines key steps."
    )
    citation2 = Citation(
        message_id=message_astro.id,
        chunk="The Planetary Society supports robust space exploration."
    )
    session.add_all([citation1, citation2])
    session.commit()
    session.refresh(citation1)
    session.refresh(citation2)

    # Add reactions to TerraBot's message
    reaction1 = Reaction(
        message_id=message_terra.id,
        reaction_name="💚",  # Green heart for Earth
        quantity=6,
        created_at=datetime.now()
    )
    reaction2 = Reaction(
        message_id=message_terra.id,
        reaction_name="🧐",  # Thinking face
        quantity=2,
        created_at=datetime.now()
    )
    session.add_all([reaction1, reaction2])
    session.commit()
    session.refresh(reaction1)
    session.refresh(reaction2)

    print("Sample data filled successfully.")


# --- Query Functions ---

def get_all_conversations(session: Session):
    """Retrieves all Conversation records."""
    return session.query(Conversation).all()


def get_all_messages(session: Session):
    """Retrieves all Message records."""
    return session.query(Message).all()


def get_all_citations(session: Session):
    """Retrieves all Citation records."""
    return session.query(Citation).all()


def get_all_reactions(session: Session):
    """Retrieves all Reaction records."""
    return session.query(Reaction).all()


# Example usage (can be run directly for basic check)
if __name__ == "__main__":
    # Create an in-memory SQLite database for this example run
    TEST_DATABASE_URL = "sqlite:///:memory:"
    test_engine = create_engine(TEST_DATABASE_URL, echo=False)  # echo=True for logging SQL

    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    print("\n--- Initializing and filling database ---")
    initialize_db(test_engine)
    test_session = TestSessionLocal()
    fill_with_sample_data(test_session)
    test_session.close()

    print("\n--- Verifying data ---")
    test_session = TestSessionLocal()
    conversations = get_all_conversations(test_session)
    messages = get_all_messages(test_session)
    citations = get_all_citations(test_session)
    reactions = get_all_reactions(test_session)

    print(f"Total Conversations: {len(conversations)}")
    print(f"Total Messages: {len(messages)}")
    print(f"Total Citations: {len(citations)}")
    print(f"Total Reactions: {len(reactions)}")

    test_session.close()

    print("\n--- Clearing all data ---")
    test_session = TestSessionLocal()
    clear_all_table_data(test_session)
    test_session.close()

    print("\n--- Verifying data cleared ---")
    test_session = TestSessionLocal()
    conversations_after_clear = get_all_conversations(test_session)
    print(f"Total Conversations after clear: {len(conversations_after_clear)}")
    test_session.close()

    print("\n--- Deleting all tables ---")
    delete_all_tables(test_engine)

    print("\n--- Attempting to access tables after deletion (should error or show empty) ---")
    # This will likely fail if tables are truly gone and you try to query without re-creation
    try:
        test_session = TestSessionLocal()
        # You'd typically need to re-initialize or catch an error here
        # For demonstration, just trying to inspect
        inspector = inspect(test_engine)
        print(f"Tables in DB after deletion: {inspector.get_table_names()}")
        test_session.close()
    except Exception as e:
        print(f"Error accessing tables after deletion: {e}")
