from together import Together
from chromadb import Client
import json


class RAGBot:
    def __init__(self, name: str, persona: str, vector_path: str, model_name: str, api_key: str):
        self.name = name
        self.persona = persona
        self.model_name = model_name
        self.client = Together(api_key=api_key)
        self.db = Client()
        self.collection = self.db.create_collection(name)

        # Load and index the vector data
        with open(vector_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                chunk = doc["chunk"]
                embedding = doc.get("embedding")

                if not embedding:
                    embedding = self.embed([chunk])[0]

                self.collection.add(
                    documents=[chunk],
                    metadatas=[{"title": doc["title"], "source": doc["id"]}],
                    ids=[doc["id"]],
                    embeddings=[embedding]
                )

    def embed(self, texts):
        result = self.client.embeddings.create(
            model="togethercomputer/m2-bert-80M-32k-retrieval",
            input=texts
        )
        return result.embeddings

    def generate(self, subject: str, chat_history: list):
        # Retrieve relevant context from the vector DB
        result = self.collection.query(
            query_texts=[subject],
            n_results=5
        )
        context = "\n".join(result["documents"][0])

        # Build the prompt
        messages = [
            {"role": "system",
             "content": f"{self.persona}\nHere is some historical context relevant to the subject:\n{context}"},
            *chat_history,
            {"role": "user", "content": f"What do you think about '{subject}'?"}
        ]

        # Generate the response
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )

        return response.choices[0].message.content
