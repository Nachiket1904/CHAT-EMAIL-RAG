from fastembed import TextEmbedding
import numpy as np
# Load the embedding model (BAAI/bge-small-en-v1.5)
embedding_model = TextEmbedding()
print("The model BAAI/bge-small-en-v1.5 is ready to use.")

# Generate embeddings for the query
def generate_query_embedding(query: str):
    embedding = list(embedding_model.embed([query]))
    return np.array(embedding)  # Convert to NumPy array for FAISS compatibility
