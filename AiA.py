from Util import load_embeddings, rerank, vector_retreival
from default_values_prompts import bot_1_name,bot_2_name,bot_1_knowledge_base,bot_2_knowledge_base
import numpy as np


class Bot:
    def __init__(self, name, persona_prompt, model, client, chat_color='#27a348', knowledge_base: str = None):
        self.name = name
        self.persona_prompt = persona_prompt
        self.model = model
        self.client = client
        self.history = []
        self.chat_color = chat_color
        if name == bot_1_name:
            knowledge_base = bot_1_knowledge_base
        elif name == bot_2_name:
            knowledge_base = bot_2_knowledge_base
        self.knowledge_base = load_embeddings(knowledge_base) if knowledge_base else None

    def clean_history(self):
        self.history = []
        return

    def generate_response(self, subject: str, user_prompt: str = None, use_knowledge: bool = True, top_k: int = 5,
                          cite=False):
        system_prompt = (f"Continue the conversation naturally.Be conversational, as if you were chatting with a "
                         f"friend Use logical connections and comparisons when changing topic.Use less than 150 "
                         f"words.Be conversational and ask the user their opinion.")
        system_messages = [{"role": "system", "content": self.persona_prompt}]
        system_messages.append({"role": "system", "content": f"Topic: {subject}" + system_prompt})
        reranked_chunks = ''
        if use_knowledge and self.knowledge_base:
            chunks = [d["chunk"] for d in self.knowledge_base]
            embeddings = np.array([d["embedding"] for d in self.knowledge_base])

            top_k_indices = vector_retreival(client=self.client, query=subject, top_k=top_k, vector_index=embeddings)
            top_k_chunks = [chunks[i] for i in top_k_indices]

            reranked_indices = rerank(self.client, chunks=top_k_chunks, top_k=top_k, query=subject)

            if cite:
                reranked_articles = [self.knowledge_base[i] for i in reranked_indices]
                stringed_articles = [f"{article['title']}\nBy:{article['author']}\n{article['chunk']}\n" for article in
                                     reranked_articles]

                reranked_chunks = "\n\n".join(stringed_articles)
                rag_prompt = (
                        "Use the following context extracted from NYT interviews to inform your next response. "
                        "Reference it only if it's relevant to the topic and always cite the tittle and author:\n\n"
                        + reranked_chunks.strip()
                )
            else:
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
            messages=messages
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})

        return {"reply": reply, "chunks": reranked_chunks.strip()}
