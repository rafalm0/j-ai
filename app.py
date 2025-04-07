from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from together import Together
from resp_evaluator import Evaluator
import uuid
from keys import api_key

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
                {"role": "system", "content": f"Talk to the user while interviewing them for: {', '.join(user_data)}."}
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


@app.post("/submit/{session_id}")
async def submit_data(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    extracted_data = sessions[session_id]["data"]
    # Save extracted_data to database (future step)
    return {"session_id": session_id, "data": extracted_data, "status": "submitted"}
