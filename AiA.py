from Util import load_embeddings, rerank, vector_retreival
import numpy as np


class Bot:
    def __init__(self, name, persona_prompt, model, client, chat_color='#27a348', knowledge_base: str = None):
        self.name = name
        self.persona_prompt = persona_prompt
        self.model = model
        self.client = client
        self.history = []
        self.chat_color = chat_color
        self.knowledge_base = None
        if knowledge_base is not None:
            self.knowledge_base = load_embeddings(knowledge_base)

    def generate_response(self, subject, use_knowledge=False, top_k=5):

        if (not use_knowledge) and (self.knowledge_base is None):
            use_knowledge = False

        situation_prompt = f" Stick to the subject: {subject}, and give your thought, continuing a chat."
        if use_knowledge:
            chunks = [d["chunk"] for d in self.knowledge_base]
            embeddings = np.array([d["embedding"] for d in self.knowledge_base])

            top_k_indices = vector_retreival(query=situation_prompt, top_k=top_k, vector_index=embeddings)
            top_k_chunks = [chunks[i] for i in top_k_indices]

            reranked_indices = rerank(situation_prompt, chunks=top_k_chunks, top_k=top_k, query=situation_prompt)
            reranked_chunks = "\n\n".join([top_k_chunks[i] for i in reranked_indices])
            situation_prompt += f"Use this info as extra source from interview of the new york times:\n\n{reranked_chunks}"

        messages = [
            {"role": "system", "content": self.persona_prompt + situation_prompt},
            *self.history
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply
