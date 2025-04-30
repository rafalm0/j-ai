class Bot:
    def __init__(self, name, persona_prompt, model, client,chat_color='#27a348'):
        self.name = name
        self.persona_prompt = persona_prompt
        self.model = model
        self.client = client
        self.history = []
        self.chat_color = chat_color

    def generate_response(self, subject):
        messages = [
            {"role": "system", "content": self.persona_prompt + f" Stick to the subject: {subject}, and give your "
                                                                f"thought, continuing a chat. Avoid big sentences."},
            *self.history
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply