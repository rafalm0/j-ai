from together import Together
import json
from pydantic import BaseModel, Field, ValidationError

class ExtractedData(BaseModel):
    age: int | None = Field(default=None)
    name: str | None = Field(default=None)
    gender: str | None = Field(default=None)

class Evaluator:
    def __init__(self, api_client, api_key,
                 model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                 extract_data=None):
        if extract_data is None:
            extract_data = ['age', 'name', 'gender']
        self.client = api_client
        self.memory = {x: None for x in extract_data}
        self.api_key = api_key
        self.model_name = model_name
        self.messages = []
        self.evaluator_logs = []

    def regen_prompt(self):
        self.prompt = f"""Extract these information from the conversation: {', '.join(self.memory.keys())}.
        If you find some or all the info, return a raw JSON object **inside $ markers**, without new lines.
        Use 'null' when something is not provided. Example:

        User: Hi, I’m Alex.
        LLM Output: ${{"name": "Alex", "age": null, "gender": null}}$

        User: I'm 25.
        LLM Output: ${{"name": "Alex", "age": 25, "gender": null}}$

        Always ensure correct JSON formatting and valid values.
        """

    def submit_message(self, message):
        self.messages.append(message)

    def evaluate(self):
        self.regen_prompt()
        messages = [{"role": "system", "content": self.prompt}] + self.messages

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )


        text_response = response.choices[0].message.content.strip()
        self.evaluator_logs.append(text_response)

        extracted_json = None
        if "$" in text_response:
            try:
                extracted_json = json.loads(text_response.split("$")[1])
            except (json.JSONDecodeError, IndexError):
                print("⚠️ Error: Could not parse JSON from response.")

        if extracted_json:
            try:
                validated_data = ExtractedData(**extracted_json).dict()
                self.memory.update({k: v for k, v in validated_data.items() if v is not None})
            except ValidationError as e:
                print(f"⚠️ Validation error: {e}")

        return self.memory
