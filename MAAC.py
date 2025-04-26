"""Multi-Agente Augmented Conversation"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from together import Together
from old_implementation.resp_evaluator import Evaluator
import uuid
from keys import api_key

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
                discussion toward extracting the users name,if they are a journalist,  how their opinion on the rise 
                of the internet and also on the rise of AI, and how these interact with journalism. Instead of 
                directly interrogating them, make the conversation engaging by discussing how the rise of the 
                internet impacted journalism and drawing parallels with the current rise of AI, particularly 
                generalist models like GPT. Share interesting facts and insights along the way, making it feel like a 
                natural discussion rather than an interview. Ask for their thoughts on the evolution of journalism 
                with the internet and AI—how they see these shifts impacting the profession. Try to weave in one or 
                two pieces of required information per message, keeping the conversation flowing smoothly with small, 
                digestible exchanges rather than overwhelming the user with too many questions at once. Once you have 
                gathered most or all of the necessary information, mention that they can click the 'People’s 
                Perception' button if they’d like to see how others feel about these changes in journalism."""}
            ],
            "data": {}
        }

    session = sessions[session_id]

    # Generate LLM response
    response = client.chat.completions.create(
        model=model_name,
        messages=session["messages"]
    )

    text_response = response.choices[0].message.content
    session["messages"].append({"role": "assistant", "content": text_response})




    return message_input.message
