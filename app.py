from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from together import Together
from resp_evaluator import Evaluator
import uuid
from keys import api_key, db_password
import psycopg2
from psycopg2.extras import RealDictCursor
import os

# Database connection settings
DB_SETTINGS = {
    "user": "rafael",
    "password": f"{db_password}",
    "host": "j-ai.postgres.database.azure.com",
    "port": 5432,
    "database": "postgres"
}

app = FastAPI()

user_data = ['name', 'age', 'gender']
model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

client = Together(api_key=api_key)
sessions = {}  # Store user sessions


class MessageInput(BaseModel):
    session_id: str | None = None
    message: str


@app.post("/chat")
async def chat(message_input: MessageInput):
    # Assign session ID
    session_id = message_input.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = {
            "evaluator": Evaluator(api_client=client, api_key=api_key),
            "messages": [
                {"role": "system", "content": f"""Talk to the user while interviewing them for: {', '.join(user_data)}. 
                The idea is to talk to persons saying interesting things abour the arrival of the internet and how it 
                compares to the arrival of AI in the field of journalist. 
                Do not simply ask questions to extract info, try to talk about the subject while also asking questions 
                to figure out the information needed. Try to get their opinion on the arrival of the internet, and the 
                arrival of the GPT/ generalists AI and how it could affect the job of the journalist. Our idea is to get
                to know the user and talk saying cool and interesting facts while also trying to extract some info. Try 
                to get maybe one or two info maximum per sentence, no need to ask all info in one question, we want small
                easy to digest conversations."""}
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
    evaluator.submit_message({"role": "user", "content": f"User: {message_input.message}"})
    evaluator.submit_message({"role": "user", "content": f"Interviewer: {text_response}"})

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
                    name TEXT,
                    age INTEGER,
                    gender TEXT,
                    is_journalist BOOLEAN,
                    years_of_practice INTEGER,
                    internet_opinion TEXT,
                    was_internet_good_for_journalist BOOLEAN,
                    gpt_opinion TEXT,
                    is_gpt_good_for_journalist BOOLEAN
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
            cursor.execute("DELETE FROM interview_data;")
            conn.commit()
    return {"message": "Database reset successfully."}


@app.post("/save-interview")
def save_interview(session_id: int):
    session = sessions[session_id]
    evaluator = session["evaluator"]
    data = evaluator.memory  # Retrieve stored memory from evaluator
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO interviews (session_id, name, age, gender, is_journalist, years_of_practice, 
                                           internet_opinion, was_internet_good_for_journalist, gpt_opinion, 
                                           is_gpt_good_for_journalist)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (session_id, data.get("name"), data.get("age"), data.get("gender"),
                      data.get("is_journalist"), data.get("years_of_practice"), data.get("internet_opinion"),
                      data.get("was_internet_good_for_journalist"), data.get("gpt_opinion"),
                      data.get("is_gpt_good_for_journalist")))
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
