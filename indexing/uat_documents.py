from openai import OpenAI
import os
from tqdm import tqdm
import pickle

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY", "")



from search_document import load_documents, search_by_id, save_documents
from prepare_query_document import prepare_query_document, extract_entities_from_titles, prepare_query_document_generic


pickle_file = 'all_documents.pickle'
documents = load_documents(pickle_file)

#documents_list = ['KB0027217']
documents_list = ['KB0027217', 'KB0026014', 'KB0025749', 'KB0026668', 'KB0024122', 'KB0014146']


def process_documents_list(documents_list):
	query_documents = {}
	doc_count = 0
	openai_client = OpenAI(api_key=openai_api_key)

	print(f"Processing documents...")

	# Process only the specified number of documents
	for doc_id in tqdm(documents_list):
		# search by document id
		doc = search_by_id(documents, doc_id)

		try:

			print(f"Doc Title {doc.get('title', '')}")

			#get content
			content = doc.get('content', '')
			#cleaned_content = clean_document_content(content)['cleaned_article']
			title = doc.get('title', ''),

			response = prepare_query_document_generic(content, title)
			related_queries = response['queries']
			#print(f"Queries ==> {related_queries}")

			#print(related_queries)

			entities = extract_entities_from_titles(title)

			for query in related_queries:
				response = openai_client.embeddings.create(
						model="text-embedding-3-small",
						input=query
					)

				embedding = response.data[0].embedding

				query_documents[query] = {
					'query_vector': embedding,
					'meta_data': {
						'entities': entities['entities'],
					},
					'query_answer': content,
					'document_title': title,
					'document_idc': doc_id
				}

		except Exception as e:
			print(f"Error in processing {doc_id}: {e}")

	return query_documents



query_documents = process_documents_list(documents_list)

#print(query_documents)

with open('query_documents_uat.pickle', 'wb') as f:
	pickle.dump(query_documents, f)


# query_doc_uat = load_documents('query_documents_uat.pickle')
# doc_ids = set()
# for query, data in query_doc_uat.items():
# 	doc_id = data.get('document_idc')
# 	print(doc_id)
# 	print(query)

