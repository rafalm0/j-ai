from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from together import Together
from resp_evaluator import Evaluator
import uuid
from keys import api_key, db_password
import psycopg2
from psycopg2.extras import RealDictCursor
import random

# Database connection settings
DB_SETTINGS = {
    "user": "rafael",
    "password": f"{db_password}",
    "host": "j-ai.postgres.database.azure.com",
    "port": 5432,
    "database": "postgres"
}

app = FastAPI()

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

user_data = ['age', 'is_journalist', 'years_of_practice', 'internet_opinion', 'internet_opinion_score', 'gpt_opinion',
             'gpt_opinion_score']
model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

client = Together(api_key=api_key)
sessions = {}  # Store user sessions


class MessageInput(BaseModel):
    session_id: str | None = None
    message: str


@app.post("/chat")
@app.options("/chat")
async def chat(message_input: MessageInput):
    session_id = message_input.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = {
            "evaluator": Evaluator(api_client=client, api_key=api_key),
            "messages": [
                {"role": "system", "content": f"""Engage in a conversation with the user while subtly guiding the 
                discussion toward extracting {', '.join(user_data)}. Instead of directly interrogating them, 
                make the conversation engaging by discussing how the rise of the internet impacted journalism and 
                drawing parallels with the current rise of AI, particularly generalist models like GPT. Share 
                interesting facts and insights along the way, making it feel like a natural discussion rather than an 
                interview. Ask for their thoughts on the evolution of journalism with the internet and AI—how they 
                see these shifts impacting the profession. Try to weave in one or two pieces of required information 
                per message, keeping the conversation flowing smoothly with small, digestible exchanges rather than 
                overwhelming the user with too many questions at once. Once you have gathered most or all of the 
                necessary information, mention that they can click the 'People’s Perception' button if they’d like to 
                see how others feel about these changes in journalism."""}
            ],
            "data": {}
        }

    session = sessions[session_id]
    evaluator = session["evaluator"]

    # Add user message
    session["messages"].append({"role": "user", "content": message_input.message})

    # Generate LLM response
    response = client.chat.completions.create(
        model=model_name,
        messages=session["messages"]
    )

    text_response = response.choices[0].message.content
    session["messages"].append({"role": "assistant", "content": text_response})

    # Send messages to evaluator for extraction
    evaluator.submit_message({"role": "user", "content": message_input.message})
    evaluator.submit_message({"role": "user", "content": text_response})

    new_data = evaluator.evaluate()
    session["data"].update(new_data)

    return {"session_id": session_id, "response": text_response, "data_collected": new_data}


@app.get("/chat/{session_id}")
async def get_chat(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"messages": sessions[session_id]["messages"]}


# --------------------------------------------------- DB CALLS -------------------------------------------------

def get_db_connection():
    return psycopg2.connect(**DB_SETTINGS, cursor_factory=RealDictCursor)


# Admin function to initialize the database
def initialize_db():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interview_data (
                    id SERIAL PRIMARY KEY,
                    age INTEGER,
                    is_journalist BOOLEAN,
                    years_of_practice INTEGER,
                    internet_opinion BOOLEAN,
                    internet_opinion_score INTEGER,
                    gpt_opinion BOOLEAN,
                    gpt_opinion_score INTEGER
                );
            """)
            conn.commit()


# Admin API endpoint to initialize the table
@app.post("/admin/init-db")
def init_db():
    initialize_db()
    return {"message": "Database initialized successfully."}


# Admin API endpoint to reset the table
@app.post("/admin/reset-db")
def reset_db():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS interview_data;")  # Drop the table
            conn.commit()
    return {"message": "Database reset successfully."}


@app.post("/save-interview/{session_id}")
def save_interview(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    data = sessions[session_id]["data"]
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO interview_data (age, is_journalist, years_of_practice, 
                                               internet_opinion, internet_opinion_score, 
                                               gpt_opinion, gpt_opinion_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (data.get("age"), data.get("is_journalist"), data.get("years_of_practice"),
                      data.get("internet_opinion"), data.get("internet_opinion_score"),
                      data.get("gpt_opinion"), data.get("gpt_opinion_score")))
                conn.commit()
        return {"message": "Interview data saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# API endpoint to fetch stored interview data
@app.get("/get-interviews")
def get_interviews():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM interview_data;")
            data = cursor.fetchall()
    return {"interviews": data}


@app.post("/admin/populate-db")
def populate_db():
    sample_data = []

    for _ in range(10):  # Generate 10 random records
        age = random.randint(18, 65)
        is_journalist = random.choice([True, False])
        years_of_practice = random.randint(0, age - 18) if is_journalist else 0

        internet_score = random.randint(-5, 5)
        internet_opinion = internet_score > 0  # True if positive, False if negative

        gpt_score = random.randint(-5, 5)
        gpt_opinion = gpt_score > 0  # True if positive, False if negative

        sample_data.append((age, is_journalist, years_of_practice,
                            internet_opinion, internet_score, gpt_opinion, gpt_score))

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany('''
                    INSERT INTO interview_data (age, is_journalist, years_of_practice, 
                                                internet_opinion, internet_opinion_score, 
                                                gpt_opinion, gpt_opinion_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', sample_data)
                conn.commit()
        return {"message": "Database populated with sample data."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
