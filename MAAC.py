from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from Util import vector_retreival,rerank, load_embeddings
from together import Together
from AiA import Bot  # This is your Bot class file
from keys import api_key

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
model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# Define your bots
bot_1999 = Bot(
    name="1999 Bot",
    persona_prompt="You are a informative journalist that understand the views of journalist from 1999 regarding the "
                   "internet. Try to make parallels to things that happened in 1999 near the y2k bug, the internet "
                   "bubble and connect with facts that ended up being true or false.",
    model=model_name,
    client=client,
    chat_color="#D0F0FD",
    knowledge_base='RAG-embeddings/nyt_1999_embedded.jsonl'
)

bot_2024 = Bot(
    name="2024 Bot",
    persona_prompt="You are a informative journalist that understand the views of journalist from 2024 regarding the "
                   "old arrival of the internet and the current arrival of the Generative AI. Try to maintain a "
                   "conversation and whenever needed try to make parrallels  on the subject to things that are "
                   "happenning with the rise of AI.",
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
    topic = session["topic"]
    turn = session["turn"]

    # Decide which bot speaks next
    current_bot = bot_1999 if turn % 2 == 0 else bot_2024

    # If it's not the first turn, pass the last message as input
    if session["history"]:
        last_message = session["history"][-1]["message"]
        current_bot.history.append({"role": "user", "content": last_message})

    # Generate bot response
    response = current_bot.generate_response(topic)
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
