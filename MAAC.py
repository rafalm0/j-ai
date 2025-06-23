from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import os
import sys
from datetime import datetime
from typing import Annotated, Generator

# Import your database and model components
from sqlalchemy.orm import Session
from db import initialize_db,get_db,Conversation, Message, Citation, Reaction

# Import your actual Bot class and Together client
from together import Together
from AiA import Bot  # Assuming your AiA.py is renamed to AIA.py

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

# Define your bots
bot_1999 = Bot(
    name="1999 Bot",
    persona_prompt="You are an informative journalist who understands the views of journalists from 1999 regarding "
                   "the internet. Try to maintain a concise, conversational tone. Make connections to key events from "
                   "that time, such as the Y2K bug, the dot-com bubble, and the public reaction to the early "
                   "internet. When appropriate, reflect on which concerns or expectations turned out to be true or "
                   "false, based on your 1999 perspective.",
    model=model_name,
    client=client,
    chat_color="#D0F0FD",
    knowledge_base='RAG-embeddings/nyt_1999_embedded.jsonl'
)

bot_2024 = Bot(
    name="2024 Bot",
    persona_prompt="You are an informative journalist who understands the views of journalists in 2024 regarding both "
                   "the early arrival of the internet and the current rise of generative AI. Maintain a "
                   "conversational and concise tone. Make relevant parallels between the internet’s emergence and "
                   "today’s AI developments, especially when discussing concerns, optimism, or societal shifts. Share "
                   "insights that help contrast how things are unfolding with AI today versus how they unfolded with "
                   "the internet.",
    model=model_name,
    client=client,
    chat_color="#C1F0C1",
    knowledge_base='RAG-embeddings/nyt_2024_embedded.jsonl'
)

# Store conversations per session
sessions = {}


class ChatInput(BaseModel):
    session_id: str | None = None
    topic: str
    continue_conversation: bool = False  # If False, we start a new conversation
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


@app.post("/multi-agent-chat")
async def multi_agent_chat(input_data: ChatInput):
    session_id = input_data.session_id or str(uuid.uuid4())

    # If new conversation, reset histories
    if not input_data.continue_conversation or session_id not in sessions:
        bot_1999.history = []
        bot_2024.history = []
        sessions[session_id] = {
            "turn": 0,
            "topic": input_data.topic,
            "history": []
        }
    session = sessions[session_id]
    if input_data.topic:
        session['topic'] = input_data.topic

    topic = session["topic"]
    turn = session["turn"]
    cite = input_data.cite

    # Decide which bot speaks next
    current_bot = bot_1999 if turn % 2 == 0 else bot_2024

    # If it's not the first turn, pass the last message as input
    if session["history"]:
        last_message = session["history"][-1]["message"]
        current_bot.history.append({"role": "user", "content": last_message})

    # Generate bot response
    response = current_bot.generate_response(topic, cite=cite)
    session["history"].append({
        "bot": current_bot.name,
        "message": response
    })

    # Increment turn
    session["turn"] += 1

    return {
        "session_id": session_id,
        "bot_name": current_bot.name,
        "response": response,
        "full_conversation": session["history"],
        "chat_color": current_bot.chat_color
    }
