import outlines
from together import Together
from pydantic import BaseModel
from typing import Optional


# Define structured output schema
class UserData(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None


class Evaluator:
    def __init__(self, api_client, api_key,
                 model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        self.client = api_client
        self.memory = UserData()  # Use Pydantic model
        self.api_key = api_key
        self.model_name = model_name
        self.messages = []

        # Initialize Outlines model
        self.llm = outlines.models.transformers(self.model_name, device="cuda")

        # Create JSON generator
        self.generator = outlines.generate.json(self.llm, UserData)

    def regen_prompt(self):
        self.prompt = """You are an AI assistant extracting user information.
        Extract the following fields from the conversation: name, age, gender.
        If information is missing, return None for that field.
        Always return a valid JSON object of type UserData.

        Conversation history:
        {conversation}

        RESULT: """  # This ensures we always get structured JSON

    def submit_message(self, message):
        self.messages.append(message)

    def evaluate(self):
        self.regen_prompt()
        conversation = "\n".join([msg["content"] for msg in self.messages])
        prompt = self.prompt.format(conversation=conversation)

        # Generate structured JSON response
        result = self.generator([prompt])[0]

        # Update memory
        self.memory = result
        return self.memory.dict()
