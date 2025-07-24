from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
# for run_cloud.py
model_save_path = "./rag/models/all-MiniLM-L6-v2"

# for running unified_local_rag 
#model_save_path = "./models/all-MiniLM-L6-v2"
model.save(model_save_path)