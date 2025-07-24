from search_document import load_documents, search_by_id, save_documents


def find_queries_for_document(query_documents, target_document_id):
    """
    Find all queries that refer to a specific document ID
    """
    matching_queries = []
    
    for query, data in query_documents.items():
        # Check if this query's document matches our target
        if data.get('document_idc') == target_document_id:
            matching_queries.append({
                'query': query,
                'data': data
            })
    
    return matching_queries



def get_original_content(original_documents, document_id):
    if isinstance(original_documents, dict):
        if document_id in original_documents:
            doc = original_documents[document_id]
            return doc.get('content', '') if isinstance(doc, dict) else str(doc)

    return None

def update_by_document_id(query_documents, target_document_id, new_content, query_document_path):
    """
    Update all queries that reference a specific document_id with new content
    """
    
    for query, data in query_documents.items():
        if data.get('document_idc') == target_document_id:
            old_content = data.get('query_answer', '')
            data['query_answer'] = new_content
    
    save_documents(query_documents, query_document_path, backup=False)


'''
Laptop Freezing - KB0026014
Laptop Hardware - KB0027217

Password Reset via Recovery Key for Mac - KB0025749
Common Outlook Issues - KB0026668
Troubleshooting: Microsoft Outlook - KB0024122
Zoom Outlook Add-in - Schedule a Meeting - KB0014146
'''


# Usage examples:
if __name__ == "__main__":
	backup = True
	query_document_path = "query_documents_openai.pickle"
	original_documents_path = "all_documents.pickle"

	query_documents = load_documents(query_document_path)

	target_document_id = "KB0014146"

	original_documents = load_documents(original_documents_path)
	new_content = get_original_content(original_documents, target_document_id)
	#print(new_content)

	if new_content:
		update_by_document_id(query_documents, target_document_id, new_content, query_document_path)

		# view modified content
		query_documents = load_documents(query_document_path)
		queries = find_queries_for_document(query_documents, target_document_id)

		print(queries)
