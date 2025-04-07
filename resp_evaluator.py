from together import Together
import json


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
        self.prompt = f"""Extract these information from the conversation : {','.join(list(self.memory.keys()))}.
        If you find some or all the info, reply like you would structure a raw JSON text, no line skip but put the JSON between $ sings so I can parse your message. 
        Use 'null' when something was not provided or not known so i can easily parse your message.
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
        try:
            parsed_response = json.loads(text_response)  # Use json.loads() instead of ast.literal_eval()
            self.memory.update(parsed_response)
        except json.JSONDecodeError:
            if "$" in text_response:
                try:
                    parsed_response = json.loads(text_response.split("$")[1])  # Use json.loads() instead of ast.literal_eval()
                    self.memory.update(parsed_response)
                except json.JSONDecodeError:
                    print("⚠️ Error: LLM response is not valid JSON")

        return self.memory
