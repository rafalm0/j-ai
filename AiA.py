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
        self.knowledge_base = load_embeddings(knowledge_base) if knowledge_base else None

    def clean_history(self):
        self.history = []
        return

    def generate_response(self, subject: str, user_prompt: str = None, use_knowledge: bool = True, top_k: int = 5):
        system_messages = [{"role": "system", "content": self.persona_prompt},
                           {"role": "system", "content": f"Topic: {subject}. Continue the conversation naturally, "
                                                         f"be brief and conversational."}]

        if use_knowledge and self.knowledge_base:
            chunks = [d["chunk"] for d in self.knowledge_base]
            embeddings = np.array([d["embedding"] for d in self.knowledge_base])

            top_k_indices = vector_retreival(query=subject, top_k=top_k, vector_index=embeddings)
            top_k_chunks = [chunks[i] for i in top_k_indices]

            reranked_indices = rerank(self.client, chunks=top_k_chunks, top_k=top_k, query=subject)
            reranked_chunks = "\n\n".join([top_k_chunks[i] for i in reranked_indices])

            rag_prompt = (
                    "Use the following context extracted from NYT interviews to inform your next response. "
                    "Reference it only if it's relevant to the topic:\n\n" + reranked_chunks.strip()
            )
            system_messages.append({"role": "system", "content": rag_prompt})

        if user_prompt:
            self.history.append({"role": "user", "content": user_prompt})

        messages = system_messages + self.history

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=250
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply
