import pickle
import numpy as np
import faiss



# 1. Load your stored query vectors
with open('query_documents_openai.pickle', 'rb') as f:
    query_documents = pickle.load(f)


# 3. Extract vectors and metadata
queries = list(query_documents.keys())
vectors = np.array([query_documents[query]['query_vector'] for query in queries], dtype=np.float32)

# 4. Create FAISS index
dimension = vectors.shape[1]  # Vector dimension
index = faiss.IndexFlatIP(dimension)
# Normalize vectors for cosine similarity
faiss.normalize_L2(vectors)
# Add vectors to the index
index.add(vectors)

faiss.write_index(index, "query_vectors_openai.index")