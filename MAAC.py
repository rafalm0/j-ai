from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import os
import sys
from datetime import datetime
from typing import Annotated, Generator

from sqlalchemy.orm import Session
from sqlalchemy import Select, Join
from db import initialize_db, get_db, Conversation, Message, Citation, Reaction

from default_values_prompts import bot_2_system, bot_2_persona, bot_2_name, bot_1_system, bot_1_persona, bot_1_name, \
    bot_1_color, bot_2_color
from together import Together
from AiA import Bot

if os.path.exists("keys.py"):
    from keys import api_key
else:
    api_key = os.environ['API_KEY']

# Initialize
app = FastAPI()

# Allow CORS
origins = [
    "http://localhost",
    "*",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Together API client
client = Together(api_key=api_key)
model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


class ChatInput(BaseModel):
    session_id: str | None = None  # if None we are stating a new convo
    topic: str
    cite: bool = False


# Event handler to initialize the database on startup
@app.on_event("startup")
async def startup_event():
    """
    This function is executed when the FastAPI application starts.
    It ensures that all database tables are created.
    """
    print("FastAPI app starting up. Initializing database...")
    initialize_db()  # Call initialize_db from database.py
    print("Database initialization complete.")


def get_or_create_conversation(
        session: Session = get_db(),
        conv_id: str = '',
        conv_name: str = 'bot_chat',
        bot_1_name: str = bot_1_name,
        bot_1_persona: str = bot_1_persona,
        bot_1_system: str = bot_1_system,
        bot_1_color: str = bot_1_color,
        bot_2_color: str = bot_2_color,
        bot_2_name: str = bot_2_name,
        bot_2_persona: str = bot_2_persona,
        bot_2_system: str = bot_2_system
):
    """
    Checks if a conversation with the given id exists. If it does, returns it.
    If not, it creates a new conversation with the provided details and returns its ID.

    Args:
        session: The SQLAlchemy session to use for database operations.
        conversation_name: The name of the conversation to find or create.
        bot_1_name, bot_1_persona, bot_1_system: Details for bot 1.
                                                  These are optional for finding an existing
                                                  conversation, but REQUIRED if a new
                                                  conversation needs to be created.
        bot_2_name, bot_2_persona, bot_2_system: Details for bot 2.
                                                  These are optional for finding an existing
                                                  conversation, but REQUIRED if a new
                                                  conversation needs to be created.

    Returns:
        The integer ID of the found or newly created conversation.

    Raises:
        ValueError: If a new conversation needs to be created but required bot details
                    (name, persona, system) are missing for either bot.
                    :param conv_name:
                    :param bot_2_color:
                    :param bot_1_color:
                    :param conv_id:
                    :param session:
                    :param bot_2_system:
                    :param bot_2_persona:
                    :param bot_2_name:
                    :param bot_1_system:
                    :param bot_1_persona:
                    :param bot_1_name:
    """

    if conv_id is not None:
        existing_conversation = session.execute(
            Select(Conversation).where(Conversation.id == conv_id)
        ).scalars().first()
        if existing_conversation:
            print(f"Found existing conversation: '{conv_id}'")
            return existing_conversation

    print(f"Conversation not found. Creating a new one...")

    new_conversation = Conversation(
        conversation_name=conv_name,
        bot_1_name=bot_1_name,
        bot_1_persona=bot_1_persona,
        bot_1_system=bot_1_system,
        bot_1_color=bot_1_color,
        bot_2_color=bot_2_color,
        bot_2_name=bot_2_name,
        bot_2_persona=bot_2_persona,
        bot_2_system=bot_2_system
    )
    session.add(new_conversation)
    session.commit()
    session.refresh(new_conversation)

    print(f"Created new conversation: '{new_conversation.conversation_name}' (ID: {new_conversation.id})")
    return new_conversation


def recover_messages_from_conversation(conversation: Conversation, get_next_bot=False):
    conn = get_db()
    info = Select(Message).where(Message.conversation_id == conversation.id)
    messages = conn.execute(info).scalars().all()
    messages = sorted(messages, key=lambda msg: msg.created_at, reverse=True)

    last_writer_name = messages[0].writer

    if get_next_bot:
        if last_writer_name == conversation.bot_1_name:
            next_bot = Bot(client=client, name=bot_2_name, persona_prompt=conversation.bot_2_persona,
                           chat_color=conversation.bot_2_color, model=model_name)
        else:
            next_bot = Bot(client=client, name=bot_1_name, persona_prompt=conversation.bot_1_persona,
                           chat_color=conversation.bot_1_color, model=model_name)
        return {"messages": messages, "bot": next_bot}
    return {"messages": messages}


def add_response(
        session: Session,
        conversation_id: int,
        message_content: str,
        writer: str,
        topic: str,
        citation: str
) -> Message:
    """
    Adds a new message (response) to the database for a given conversation.

    Args:
        session: The SQLAlchemy session to use.
        conversation_id: The ID of the conversation this message belongs to.
        message_content: The actual text content of the message.
        writer: The name of the bot (or user) who wrote the message.
        topic: The topic associated with this message.
        citation: the chunk that was cited

    Returns:
        The newly created Message ORM object after it's committed to the DB.

    """
    new_message = Message(
        conversation_id=conversation_id,
        message=message_content,
        writer=writer,
        topic=topic,
        created_at=datetime.now()  # Use current time for the new message
    )

    session.add(new_message)
    session.commit()
    session.refresh(new_message)  # Get the ID and any default values assigned by DB
    print(f"Added new message to conversation {conversation_id} by {writer}: {message_content[:50]}...")
    citation = Citation(message_id=new_message.id, chunk=citation)
    session.add(citation)
    session.commit()
    print(f"Added new citation to conversation {conversation_id}: {citation[:50]}...")
    return new_message


def build_bot_from_conversation(conversation: Conversation, bot_name=None):
    if (bot_name == conversation.bot_1_name) or (bot_name is None):
        bot = Bot(client=client, name=conversation.bot_1_name, persona_prompt=conversation.bot_1_persona,
                  chat_color=conversation.bot_1_color, model=model_name)
    else:
        bot = Bot(client=client, name=conversation.bot_2_name, persona_prompt=conversation.bot_2_persona,
                  chat_color=conversation.bot_2_color, model=model_name)
    return bot


@app.post("/multi-agent-chat")
async def multi_agent_chat(input_data: ChatInput):
    conversation_id = input_data.session_id

    conversation = get_or_create_conversation(conv_id=conversation_id)
    if conversation.id == conversation_id:
        aux = recover_messages_from_conversation(conversation, get_next_bot=True)
    else:
        aux = {"messages": [], "bot": build_bot_from_conversation(conversation, conversation.bot_1_name)}
    messages = aux['messages']
    next_bot = aux['bot']
    topic = input_data.topic
    cite = input_data.cite
    response = next_bot.generate_response(subject=topic, cite=cite)
    reply_response = response['reply']
    chunks = response['chunks']
    add_response(get_db(), int(conversation.id), message_content=reply_response, writer=next_bot.name, topic=topic,
                 citation=chunks)

    history = [
        {"name": msg.writer, "content": msg.message}
        for msg in messages
    ]
    # history -> list like this: [{"name": "bot_1", "content": "hello"},{"name": "bot_2", "content": "hello_there"}]
    return {
        "session_id": conversation.id,
        "bot_name": next_bot.name,
        "response": reply_response,
        "full_conversation": history,
        "chat_color": next_bot.chat_color
    }
