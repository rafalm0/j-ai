from together import Together
import json
from pydantic import BaseModel, Field, ValidationError


class ExtractedData(BaseModel):
    age: int | None = Field(default=None)
    is_journalist: bool | None = Field(default=None)
    years_of_practice: int | None = Field(default=None)
    internet_opinion: bool | None = Field(default=None)  # True if positive, False if negative
    internet_opinion_score: int | None = Field(default=None)  # Score from -5 to 5
    gpt_opinion: bool | None = Field(default=None)  # True if positive, False if negative
    gpt_opinion_score: int | None = Field(default=None)  # Score from -5 to 5


class Evaluator:
    def __init__(self, api_client, api_key,
                 model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                 extract_data=None):
        if extract_data is None:
            extract_data = ['age', 'is_journalist', 'years_of_practice', 'internet_opinion',
                            'internet_opinion_score', 'gpt_opinion', 'gpt_opinion_score']
        self.client = api_client
        self.memory = {x: None for x in extract_data}
        self.api_key = api_key
        self.model_name = model_name
        self.messages = []
        self.evaluator_logs = []

    def regen_prompt(self):
        self.prompt = """Extract the following information from the conversation as a JSON object inside `$` markers.
        Ensure the JSON is properly formatted with the correct types:

        - age: integer or null (e.g., 25 or null)
        - is_journalist: boolean or null (true if the person is a journalist, otherwise false or null)
        - years_of_practice: integer or null (number of years, or null if unknown)
        - internet_opinion: boolean or null (true if positive, false if negative)
        - internet_opinion_score: integer or null (sentiment score from -5 to 5)
        - gpt_opinion: boolean or null (true if positive, false if negative)
        - gpt_opinion_score: integer or null (sentiment score from -5 to 5)

        Example conversations:

        User: Hi, I’m Alex. 
        LLM Output: ${"age": null, "is_journalist": null, "years_of_practice": null, 
                      "internet_opinion": null, "internet_opinion_score": null, 
                      "gpt_opinion": null, "gpt_opinion_score": null}$

        User: I'm 25 and I've been a journalist for 5 years.
        LLM Output: ${"age": 25, "is_journalist": true, "years_of_practice": 5, 
                      "internet_opinion": null, "internet_opinion_score": null, 
                      "gpt_opinion": null, "gpt_opinion_score": null}$

        User: I think the internet was great for journalism, but GPT might be harmful.
        LLM Output: ${"age": 25, "is_journalist": true, "years_of_practice": 5, 
                      "internet_opinion": true, "internet_opinion_score": 4, 
                      "gpt_opinion": false, "gpt_opinion_score": -3}$

        **DO NOT** return any extra text, only the JSON object inside `$` markers.
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
