import uuid
import re
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from sentence_transformers import util

load_dotenv()

# Initialize SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Normalize text for consistent embeddings
def normalize(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

# Generate embeddings from text
def get_embedding(text):
    try:
        return embedding_model.encode(normalize(text)).tolist()
    except Exception as e:
        print(f"❌ Error getting embedding: {str(e)}")
        raise

from typing import List

# Embedding function wrapper for Chroma
class ChromaEmbeddingFunction:
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return [get_embedding(text) for text in texts]

# Initialize ChromaDB client and collection with cosine similarity
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(
    name="chatbot_memory",
    metadata={"hnsw:space": "cosine"},
)

# Rerank retrieved results using embedding similarity
def rerank_memory(query, memory_results):
    if not memory_results:
        return []

    query_emb = embedding_model.encode(normalize(query), convert_to_tensor=True)
    doc_texts = [item["document"] for item in memory_results]
    doc_embs = embedding_model.encode(doc_texts, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(query_emb, doc_embs)[0]
    ranked_results = sorted(zip(memory_results, scores), key=lambda x: x[1], reverse=True)

    return [
        {
            **item,
            "rerank_score": float(score)
        } for item, score in ranked_results
    ]


# Check if similar memory already exists
def already_exists(embedding, threshold=0.90):
    try:
        results = collection.query(query_embeddings=[embedding], n_results=1)
        if not results["documents"][0]:
            return False
        score = results["distances"][0][0]  # Cosine distance: smaller is better
        return score < (1.0 - threshold)
    except Exception as e:
        print(f"❌ Error checking duplicates: {str(e)}")
        return False

# Store memory if not already present
def store_memory(prompt, sql, result, final_response):
    try:
        combined_text = f"User Prompt: {prompt}\nSQL Query: {sql}\nSQL Result: {result}\nFinal Response: {final_response}"
        embedding = get_embedding(combined_text)

        if already_exists(embedding):
            print("⚠️ Similar memory already exists. Skipping storage.")
            return

        collection.add(
            documents=[combined_text],
            embeddings=[embedding],
            metadatas=[{
                "source": "chat_interaction",
                "prompt": prompt,
                "sql": sql,
                "result": str(result),
                "response": final_response
            }],
            ids=[str(uuid.uuid4())]
        )
        print("✅ Stored embedding in ChromaDB.")
    except Exception as e:
        print(f"❌ Error storing memory: {str(e)}")

# Retrieve relevant past interactions
def retrieve_memory(query, top_k=5):
    try:
        query_embedding = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        memory_results = [
            {
                "document": doc,
                "distance": dist,
                "id": doc_id,
                "metadata": meta
            }
            for doc, dist, doc_id, meta in zip(documents, distances, ids, metadatas)
        ]

        print(f"🔎 Retrieved {len(memory_results)} similar memories:")
        for i, item in enumerate(memory_results):
            print(f"  {i+1}. {item['document'][:80]}... (distance: {item['distance']:.4f})")

        return rerank_memory(query, memory_results)

    except Exception as e:
        print(f"❌ Error retrieving memory: {str(e)}")
        return []
