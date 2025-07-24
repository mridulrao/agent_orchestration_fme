'''
Should return query_document relevant to the user 

Search for user_query comparison to 
'''

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. Load your stored query vectors
with open('query_documents.pickle', 'rb') as f:
    query_documents = pickle.load(f)

# 2. Create a model for encoding new queries (use the same model you used before)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your actual embedding model

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

# 5. Function to perform similarity search using FAISS
def find_similar_queries(user_query, query_documents, queries, index, top_k=5):
    # Convert user query to vector
    user_query_vector = model.encode([user_query])[0]
    # Reshape and normalize
    user_query_vector = np.array([user_query_vector], dtype=np.float32)
    faiss.normalize_L2(user_query_vector)
    
    # Search
    scores, indices = index.search(user_query_vector, top_k)
    
    # Return results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(queries):
            original_query = queries[idx]
            doc_data = query_documents[original_query]
            results.append({
                'query': original_query,
                'query_answer': doc_data['query_answer'],
                'document_title': doc_data['document_title'],
                'document_id': doc_data['document_idc'],
                'entities': doc_data['meta_data']['entities'],
                'similarity_score': float(scores[0][i])
            })
    
    return results

# Example usage
user_input = input("Enter your query: ")
similar_queries = find_similar_queries(user_input, query_documents, queries, index)
for i, result in enumerate(similar_queries):
    print(f"Result {i+1}: Query: {result['query']}")
    print(f"Document: {result['document_title']}")
    print(f"Document ID: {result['document_id']}")
    print(f"Answer: {result['query_answer']}")
    print(f"Similarity Score: {result['similarity_score']:.4f}")
    print("---")