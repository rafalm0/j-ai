from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from datetime import datetime

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


# --------------------------------------- Inputs -------------------------------------------------------------------

class ChatInput(BaseModel):
    session_id: str | None = None  # if None we are stating a new convo
    topic: str
    cite: bool = False
    conv_name: str | None = None
    bot_1_name: str | None = None
    bot_2_name: str | None = None


class ReactionInput(BaseModel):
    message_id: str
    emoji: str | None = None


class ConversationInput(BaseModel):
    conv_id: str


class MessageInput(BaseModel):
    message_id: str


# --------------------------------------- Startup -------------------------------------------------------------------
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


# --------------------------------------- Functions -------------------------------------------------------------------
def get_or_create_conversation(
        session: Session = get_db(),
        conv_id: int = None,
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
        conv_name: The name of the conversation to find or create.
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


def recover_messages_from_conversation(conversation: Conversation, get_next_bot=False, get_reacts=False):
    conn = get_db()
    info = Select(Message).where(Message.conversation_id == conversation.id)
    messages = conn.execute(info).scalars().all()
    messages = sorted(messages, key=lambda msg: msg.created_at, reverse=True)

    last_writer_name = messages[0].writer
    print(f"Last writer detected: {last_writer_name}, finding next one...")

    result = {}

    message_result = []
    color = {True: "#D0F0FD", False: "#C1F0C1"}
    flag = True
    message_finder = {}
    for i, message in enumerate(messages):  # convert message into dict with same structure
        message_finder[message.id] = i
        message_result.append({"message_id": message.id, "text": message.message, "bot": message.writer,
                               "chat_color": color[flag], "reacts": []})
        flag = not flag

    if get_reacts:
        info = Select(Reaction).join(Message, Reaction.message_id == Message.id).where(
            Message.conversation_id == conversation.id)
        reacts = conn.execute(info).scalars().all()

        for react in reacts:  # add the reacts to respective message
            message_result[message_finder[react.message_id]]['reacts'].append(
                {"reaction": react.reaction_name, "quantity": react.quantity})

    if get_next_bot:
        if last_writer_name == conversation.bot_1_name:
            print(f"Matched bot 1: {conversation.bot_1_name}")
            next_bot = build_bot_from_conversation(conversation, conversation.bot_2_name)
        elif last_writer_name == conversation.bot_2_name:
            print(f"Matched bot 2: {conversation.bot_2_name}")
            next_bot = build_bot_from_conversation(conversation, conversation.bot_1_name)
        else:
            print(f"[WARNING] Bot name not found, falling back to bot 1")
            next_bot = build_bot_from_conversation(conversation, conversation.bot_1_name)
        print(f"Next bot will be: {next_bot}")
        result['bot'] = next_bot
    result['messages'] = message_result
    return result


def getall_conversations():
    conn = get_db()
    info = Select(Conversation)
    conversations = conn.execute(info).scalars().all()
    return {"conversations": conversations}


def get_conversation(conversation_id: int):
    conn = get_db()
    info = Select(Conversation).where(Conversation.id == conversation_id)
    conversations = conn.execute(info).scalars().all()
    if len(conversations) == 0:
        return {"Message": "[Error] Conversation not found"}
    elif len(conversations) > 1:
        print(f"[WARNING] More than one conversation matched on id {conversation_id}")
        return {"Message": "[Error] Conversation not found"}
    else:
        conv = conversations[0]
    return {"Message": "Retrival successful", "conversation": conv}


def add_response(
        conversation_id: int,
        message_content: str,
        writer: str,
        topic: str,
        citation: str
) -> Message:
    """
    Adds a new message (response) to the database for a given conversation.

    Args:
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
    session = get_db()
    session.add(new_message)
    session.commit()
    session.refresh(new_message)  # Get the ID and any default values assigned by DB
    print(f"Added new message to conversation {conversation_id} by {writer}: {message_content[:50]}...")
    citation = Citation(message_id=new_message.id, chunk=citation)
    session = get_db()
    session.add(citation)
    session.commit()
    print(f"Added new citation to conversation {conversation_id}: {citation.chunk[:50]}...")
    return new_message


def build_bot_from_conversation(conversation: Conversation, bot_name=None):
    if (bot_name == conversation.bot_1_name) or (bot_name is None):
        bot = Bot(client=client, name=conversation.bot_1_name, persona_prompt=conversation.bot_1_persona,
                  chat_color=conversation.bot_1_color, model=model_name)
    else:
        bot = Bot(client=client, name=conversation.bot_2_name, persona_prompt=conversation.bot_2_persona,
                  chat_color=conversation.bot_2_color, model=model_name)
    print(f"New bot constructed to reply: {bot.name}")
    return bot


def react_emoji(message: Message.id, emoji):
    emoji_exist = False
    conn = get_db()
    info = Select(Reaction).where(Reaction.message_id == message).where(Reaction.reaction_name == emoji)
    reacts = conn.execute(info).scalars().all()

    if len(reacts) == 1:
        emoji_exist = True
    elif len(reacts) != 0:
        print(
            f"[WARNING] More than one log of reaction {emoji} found in message with id {message}, skipping reaction...")

    edited_reaction = None
    if emoji_exist:
        react = reacts[0]
        react.quantity += 1
        conn.commit()
        print(f"Added +1 reaction to reaction {react.id}: {emoji}...")
        edited_reaction = react
    else:
        new_reaction = Reaction(
            message_id=message,
            reaction_name=emoji,
            quantity=1,
            created_at=datetime.now()  # Use current time for the new reaction
        )
        session = get_db()
        session.add(new_reaction)
        session.commit()
        session.refresh(new_reaction)
        print(f"Added new reaction to message {message}: {emoji}...")
        edited_reaction = new_reaction
    return edited_reaction


def get_emojis(message: Message.id):
    conn = get_db()
    info = Select(Reaction).where(Reaction.message_id == message)
    reacts = conn.execute(info).scalars().all()

    return reacts


def remove_message(message_id: int):
    """
    Removes a message
    If the message exists, it is deleted from the database.
    If it doesn't exist, nothing happens.
    """
    conn = get_db()
    info = Select(Message).where(Message.id == message_id)
    messages = conn.execute(info).scalars().all()

    if len(messages) == 1:
        message = messages[0]
        conn.delete(message)
        conn.commit()
        print(f"Deleted message {message}")
        return {"status": "deleted", "message_id": message_id}
    elif len(messages) > 1:
        print(f"[WARNING] Multiple messages found with id {message_id}, skipping delete...")
        return {"status": "error", "reason": "duplicate reactions"}
    else:
        print(f"No message found with id: {message_id}")
        return {"status": "not_found"}


def remove_emoji_reaction(message_id: int, emoji: str):
    """
    Removes an emoji reaction from a specific message.
    If the reaction exists, it is deleted from the database.
    If it doesn't exist, nothing happens.
    """
    conn = get_db()
    info = Select(Reaction).where(Reaction.message_id == message_id).where(Reaction.reaction_name == emoji)
    reacts = conn.execute(info).scalars().all()

    if len(reacts) == 1:
        react = reacts[0]
        conn.delete(react)
        conn.commit()
        print(f"Deleted reaction {emoji} from message {message_id}")
        return {"status": "deleted", "reaction_id": react.id}
    elif len(reacts) > 1:
        print(f"[WARNING] Multiple reactions found for {emoji} on message {message_id}, skipping delete...")
        return {"status": "error", "reason": "duplicate reactions"}
    else:
        print(f"No reaction {emoji} found on message {message_id}")
        return {"status": "not_found"}


def clear_emojis(message_id: int):
    """
    Removes all emojis reaction from a specific message.
    """
    conn = get_db()
    info = Select(Reaction).where(Reaction.message_id == message_id)
    reacts = conn.execute(info).scalars().all()
    for react in reacts:
        conn.delete(react)
    conn.commit()

    return {"status": "deleted"}


# --------------------------------------- Endpoints -------------------------------------------------------------------
@app.post("/multi-agent-chat")
async def multi_agent_chat(input_data: ChatInput):
    conversation_id = input_data.session_id
    if conversation_id == 'None':
        conversation_id = None
    if conversation_id is not None:
        conversation_id = int(conversation_id)

    conversation = get_or_create_conversation(conv_id=conversation_id,
                                              conv_name=input_data.conv_name,
                                              bot_1_name=input_data.bot_1_name,
                                              bot_2_name=input_data.bot_2_name)
    if conversation.id == conversation_id:
        print("Conversation matched, recovering previous messages...")
        aux = recover_messages_from_conversation(conversation, get_next_bot=True)
    else:
        print(
            f"Conversation {conversation.id} not matched with {conversation_id}, falling back to new conversation... ")
        aux = {"messages": [], "bot": build_bot_from_conversation(conversation, conversation.bot_1_name)}
    messages = aux['messages']
    next_bot = aux['bot']
    topic = input_data.topic
    cite = input_data.cite
    response = next_bot.generate_response(subject=topic, cite=cite)
    reply_response = response['reply']
    chunks = response['chunks']
    new_message = add_response(int(conversation.id), message_content=reply_response, writer=next_bot.name, topic=topic,
                               citation=chunks)

    history = [
        {"name": msg['bot'], "content": msg['text'], "message_id": msg['message_id']}
        for msg in messages
    ]
    # history -> list like this: [{"name": "bot_1", "content": "hello"},{"name": "bot_2", "content": "hello_there"}]
    return {
        "conversation_id": conversation.id,
        "bot": next_bot.name,
        "text": reply_response,
        "message_id": new_message.id,
        "full_conversation": history,
        "chat_color": next_bot.chat_color
    }


@app.post("/react")
async def reaction(input_data: ReactionInput):
    message_id = input_data.message_id
    emoji = input_data.emoji
    react_emoji(message_id, emoji)

    return {"message": "reaction logged :)"}


@app.get("/react")
async def reaction(input_data: ConversationInput):
    cov_id = input_data.covnersation_id
    reacts = get_emojis(cov_id)

    return reacts


@app.delete("/react")
async def reaction(input_data: ReactionInput):
    message_id = input_data.message_id
    emoji = input_data.emoji
    if emoji is None:
        return {"message": "[WARNING] No Emoji selected for deletion, did you mean /clearreacts?"}
    remove_emoji_reaction(int(message_id), emoji)
    return {"message": "reaction deleted"}


@app.delete("/clearreacts")
async def reaction(input_data: ReactionInput):
    message_id = input_data.message_id
    clear_emojis(int(message_id))
    return {"message": "reaction deleted"}


@app.delete("/message")
async def del_message(input_data: MessageInput):
    response = {}

    message_id = int(input_data.message_id)
    remove_message(int(message_id))

    response['message'] = 'message deleted'
    return response


@app.get("/conversations")
async def conversations():
    convs = getall_conversations()['conversations']
    response = {"conversations": []}
    for conv in convs:
        _id = conv.id
        _name = conv.conversation_name
        _bot1 = conv.bot_1_name
        _bot2 = conv.bot_2_name
        _messages = conv.messages
        _topic = ""
        if _messages:
            _topic = _messages[0].topic
        c = {"id": _id, "name": _name, "bot1": _bot1, "bot2": _bot2, "Topic": _topic}
        response['conversations'].append(c)
    return response


@app.post("/conversation")
async def conversation(input_data: ConversationInput):
    conv = get_conversation(conversation_id=int(input_data.conv_id))
    if "conversation" in conv.keys():
        print(conv['Message'])
        conv = conv['conversation']
        return recover_messages_from_conversation(conv, get_reacts=True)
    else:
        return conv['Message']


# recover_messages_from_conversation(Conversation(id=40), get_reacts=True)
print("System started...")
