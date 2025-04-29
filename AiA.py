class Bot:
    def __init__(self, name, persona_prompt, model, client):
        self.name = name
        self.persona_prompt = persona_prompt
        self.model = model
        self.client = client
        self.history = []

    def generate_response(self, subject):
        messages = [
            {"role": "system", "content": self.persona_prompt},
            *self.history,
            {"role": "user", "content": f"Topic: {subject}. What are your thoughts?"}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply