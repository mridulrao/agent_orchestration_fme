'''
query_documents{
	'query': 
	'query_vector':
	'meta_data': {
			'entities': []
		}
	'query_answer' <answer related to that query>
	'document_title':
	'document_id':
}

'''

from openai import OpenAI, AzureOpenAI
from search_document import load_documents, search_by_id
import json
from tqdm import tqdm
import pickle
import sentence_transformers

import os

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY", "")

embedding_model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

def clean_document_content(document):
	openai_client = OpenAI(api_key=openai_api_key)

	system_prompt = '''
		You are a technical document cleaning specialist. Your task is to clean and format technical articles by removing formatting artifacts and unwanted elements while preserving all original information intact.

		Your cleaning tasks:
		1. Remove all non-breaking space characters (\xa0) and replace with regular spaces
		2. Remove all website links and URLs (including SharePoint links, file paths starting with http/https, etc.)
		3. Remove link markdown formatting like [text](url) - keep only the text part
		4. Clean up excessive whitespace and normalize spacing
		5. Remove any garbled characters or encoding artifacts
		6. Preserve all technical information, procedures, steps, and instructions exactly as written
		7. Maintain the original structure and formatting (headings, numbering, tables, etc.)
		8. Keep all technical details like file paths, server names, folder names, and commands intact

		Critical guidelines:
		- DO NOT add any new information or improve the content
		- DO NOT rephrase or rewrite any technical instructions
		- DO NOT add explanations or clarifications
		- DO NOT remove any technical details, steps, or procedures
		- DO NOT change the meaning or structure of the content
		- ONLY remove formatting artifacts, links, and clean up display issues

		Input: Raw technical article content with formatting issues
		Output: Clean, readable version of the same content with all technical information preserved

		Format your output as a JSON object:
		{
		    "cleaned_article": "cleaned content goes here"
		}

		Example transformation:
		Input: "1.0 Executive Summary\nThis article provide instructions..."
		Output JSON: {"cleaned_article": "1.0 Executive Summary\nThis article provide instructions..."}

		Input: "[LabX Document](https://gileadconnect.sharepoint.com/...)"
		Output JSON: {"cleaned_article": "LabX Document"}

		Clean the provided article content following these guidelines and return the result in the specified JSON format.
	'''

	user_prompt = f'''
		Analyze this technical document 
		==========================
		{document}
		==========================
	'''

	response = openai_client.chat.completions.create(
					model="gpt-4o",
					messages=[
						{"role": "system", "content": system_prompt},
						{"role": "user", "content": user_prompt}
						],
					temperature=0.1,
					response_format={"type": "json_object"}
				)

	extract_response = json.loads(response.choices[0].message.content)

	return extract_response

def prepare_query_document(document, document_title):
	openai_client = OpenAI(api_key=openai_api_key)

	system_prompt = '''
		You are a technical content analyzer specializing in generating relevant queries from technical documentation. Your task is to:

		1. Carefully analyze the provided technical document and its structure.
		2. Identify the main topics, problems, and concepts covered in the document.
		3. Extract or note the article title/heading for context.
		4. Generate exactly 5 practical, searchable queries that could be answered using the information in this document.
		5. Make each query rich in entities and technical terms from the document title and content to enable effective lookup-table matching.

		Important guidelines for query generation:
		- Queries should be phrased as real-world questions someone would ask when seeking this information
		- Examples: "How do I fix...", "What causes...", "How can I troubleshoot...", "What steps should I take to...", "How to configure...", "What is the process for..."
		- Include specific technical terms, product names, and entities from the document
		- Reference the document title/heading context in the queries when relevant
		- Make queries specific enough to match against this particular document
		- Ensure queries represent different aspects or sections of the document when possible

		Format your output as a JSON object with:
		- "article_title": The title or main heading of the document
		- "queries": An array of exactly 5 generated queries

		Example output format:
		{
		    "article_title": "Troubleshooting Database Connection Issues in Production Environment",
		    "queries": [
		        "How do I troubleshoot slow database connection performance in production environment?",
		        "What causes database connection timeout errors in high-traffic applications?",
		        "How can I diagnose database connection pool exhaustion issues?",
		        "What steps should I take to resolve database authentication failures?",
		        "How to monitor and optimize database connection performance metrics?"
		    ]
		}
	'''

	user_prompt = f'''
		Analyze this technical document 
		Technical Document: {document_title}
		==========================
		{document}
		==========================
	'''

	response = openai_client.chat.completions.create(
					model="gpt-4o",
					messages=[
						{"role": "system", "content": system_prompt},
						{"role": "user", "content": user_prompt}
						],
					temperature=0.1,
					response_format={"type": "json_object"}
				)

	extract_response = json.loads(response.choices[0].message.content)

	return extract_response

def prepare_query_document_generic(document, document_title):
	openai_client = OpenAI(api_key=openai_api_key)

	system_prompt = """
	You are a technical‑content analyst for an IT‑support RAG system.

	TASK
	Given one technical article:
	1. Read its main heading/title.
	2. Identify the five most important problems it addresses.
	3. For each problem create **three** types queries:
	   • **Specific**– include exact product names, tools, or procedures from the doc.  
	   • **Generic**– everyday wording that still mentions the device/OS/problem.  
	   • **Symptom**– very short, plain‑language complaint (no brand names, no “how to”).  

	OUTPUT
	Return only valid JSON:

	{
	  "article_title": "<document title>",
	  "queries": [
	  ]
	}

	RULES
	• Generic queries must map to the same problem in plain words.  
	• Symptom queries should be ≤6 words (“PC won’t start”, “System freezing”).  
	• Keep natural English question/statement forms—no bullet punctuation.  
	• JSON only—no extra commentary.

	EXAMPLE  
	====================
	INPUT title  
	“Troubleshooting Laptop Freezing on Windows 11 (KB0026014)”

	OUTPUT  
	{
	  "article_title": "Troubleshooting Laptop Freezing on Windows 11 (KB0026014)",
	  "queries": [
	    "How do I run Dell Command Update to fix laptop freezing on Windows 11?",
	    "My Windows laptop keeps freezing—how can I stop it?",
	    "Computer is frozen",
	    "How can I use Windows Hardware Diagnostics to identify faults causing my Dell laptop to freeze?",
	    "Why does my laptop hang randomly and how do I test for hardware issues?",
	    "Laptop not responding",
	    "What steps should I take to clear temporary files and cache on a Dell Latitude running Windows 11 to prevent freezing?",
	    "Could deleting temp files help if my PC keeps locking up?",
	    "System running slow",
	    "How do I perform a BIOS update on a Dell Latitude to address random system freezes?",
	    "Is updating my BIOS helpful if my laptop keeps crashing?",
	    "Laptop keeps crashing",
	    "What is the recommended driver update sequence for resolving Dell Latitude freeze issues on Windows 11?",
	    "Which drivers should I update first if my laptop keeps freezing?",
	    "Laptop keeps freezing"
	  ]
	}
	====================
	"""



	user_prompt = f'''
		Analyze this technical document 
		Technical Document: {document_title}
		==========================
		{document}
		==========================
	'''

	response = openai_client.chat.completions.create(
					model="gpt-4o",
					messages=[
						{"role": "system", "content": system_prompt},
						{"role": "user", "content": user_prompt}
						],
					temperature=0.1,
					response_format={"type": "json_object"}
				)

	extract_response = json.loads(response.choices[0].message.content)

	return extract_response


def extract_entities_from_titles(kb_article_titles):
	"""
	Extract entities from KB article titles that have a high probability of being 
	incorrectly transcribed by Speech-to-Text systems.

	Args:
	kb_article_titles (list): List of KB article titles used for generating query answers
	openai_api_key (str): OpenAI API key

	Returns:
	list: List of entities likely to be mistranscribed by STT
	"""
	# Join the titles with newline separators for better context
	titles_content = "\n".join(kb_article_titles)

	openai_client = OpenAI(api_key=openai_api_key)

	system_prompt = '''
		You are an advanced entity extraction system specialized in identifying terms from technical knowledge base article titles that have a high probability of being incorrectly transcribed by Speech-to-Text (STT) systems.

		When presented with KB article titles:

		1. Identify and extract key terms that are likely to be misheard or mistranscribed, focusing on:
		- Product names and brands (e.g., TruU, Azure AD, OAuth2)
		- Technical terms and acronyms (e.g., SSO, SAML, MFA)
		- Terms with unusual spelling, capitalization, or pronunciation
		- Version numbers and product identifiers
		- Domain-specific terminology
		- Compound technical terms (e.g., "Zero Trust Network Access")

		2. Prioritize terms that:
		- Are central to the article's topic
		- Have non-standard pronunciation or spelling
		- Contain mixed case, numbers, or special characters
		- Are technical jargon specific to the domain
		- Would be critical for correctly routing user queries

		3. Format your response as a valid JSON object with a single "entities" key containing an array of identified entities:
		{
		   "entities": ["Entity1", "Entity2", "Entity3", ...]
		}
	'''

	user_prompt = f'''
		Please analyze the following knowledge base article titles and extract key terms that have a high probability of being incorrectly transcribed by Speech-to-Text systems. Focus on technical terms, product names, and specialized vocabulary that would be critical for correctly understanding user queries about these topics.

		KB Article Titles:
		==========================
		{titles_content}
		==========================
	'''

	response = openai_client.chat.completions.create(
	    model="gpt-4o",
	    messages=[
	        {"role": "system", "content": system_prompt},
	        {"role": "user", "content": user_prompt}
	    ],
	    temperature=0.1,
	    response_format={"type": "json_object"}
	)

	extract_response = json.loads(response.choices[0].message.content)

	return extract_response


def process_documents(documents):
	query_documents = {}
	doc_count = 0

	print(f"Processing documents...")

	# Get list of document IDs
	doc_ids = list(documents.keys())

	# Process only the specified number of documents
	for doc_id in tqdm(doc_ids[:]):
		# search by document id
		doc = search_by_id(documents, doc_id)

		try:
			#get content
			content = doc.get('content', '')
			cleaned_content = clean_document_content(content)['cleaned_article']
			title = doc.get('title', ''),

			response = prepare_query_document(content, title)
			related_queries = response['queries']

			print(related_queries)

			entities = extract_entities_from_titles(title)

			for query in related_queries:
				query_vector = embedding_model.encode([query])[0]
				query_documents[query] = {
					'query_vector': query_vector,
					'meta_data': {
						'entities': entities['entities'],
					},
					'query_answer': cleaned_content,
					'document_title': title,
					'document_idc': doc_id
				}


			print(query_documents)

			# for key in response.keys():
			# 	query = key
			# 	answer = response[key]
			# 	query_vector = embedding_model.encode([query])[0]

			# 	entities = extract_entities_from_titles(title)

			# 	query_documents[query] = {
			# 	'query_vector': query_vector,
			# 	'meta_data': {
			# 	    'entities': entities['entities']
			# 	},
			# 	'query_answer': answer,
			# 	'document_title': doc.get('title', ''),
			# 	'document_id': doc.get('document_id', '')
			# 	}

		except Exception as e:
			print(f"Error in processing doc id: {doc_id}, error: {e}")

	return query_documents

def process_documents_list(documents_list):
	query_documents = {}
	doc_count = 0

	print(f"Processing documents...")

	# Process only the specified number of documents
	for doc_id in tqdm(documents_list):
		# search by document id
		doc = search_by_id(documents, doc_id)

		try:

			print(f"Doc Title {doc.get('title', '')}")

			#get content
			content = doc.get('content', '')
			cleaned_content = clean_document_content(content)['cleaned_article']
			title = doc.get('title', ''),

			response = prepare_query_document(content, title)
			related_queries = response['queries']

			#print(related_queries)

			entities = extract_entities_from_titles(title)

			for query in related_queries:
				query_vector = embedding_model.encode([query])[0]
				query_documents[query] = {
					'query_vector': query_vector,
					'meta_data': {
						'entities': entities['entities'],
					},
					'query_answer': cleaned_content,
					'document_title': title,
					'document_idc': doc_id
				}


			#print(query_documents)

			# for key in response.keys():
			# 	query = key
			# 	answer = response[key]
			# 	query_vector = embedding_model.encode([query])[0]

			# 	entities = extract_entities_from_titles(title)

			# 	query_documents[query] = {
			# 	'query_vector': query_vector,
			# 	'meta_data': {
			# 	    'entities': entities['entities']
			# 	},
			# 	'query_answer': answer,
			# 	'document_title': doc.get('title', ''),
			# 	'document_id': doc.get('document_id', '')
			# 	}

			# doc_count += 1

		except Exception as e:
			print(f"Error in processing {doc_id}: {e}")

	return query_documents




# Example usage
if __name__ == "__main__":
	# load documents 
	pickle_file = 'all_documents.pickle'
	documents = load_documents(pickle_file)

	#query_documents = process_documents(documents)
	new_kb_numbers = [
	    "KB0030058", "KB0030060", "KB0026647", "KB0026197", "KB0021049",
	    "KB0015718", "KB0028960", "KB0029404", "KB0020796", "KB0020129",
	    "KB0027909", "KB0028466", "KB0027414", "KB0020610", "KB0028655",
	    "KB0014288", "KB0027418", "KB0028924", "KB0022734", "KB0028797",
	    "KB0028796", "KB0016190", "KB0027441", "KB0029355", "KB0020565",
	    "KB0022702", "KB0027968", "KB0027416", "KB0029016", "KB0026680",
	    "KB0020714", "KB0020921", "KB0028351", "KB0028643", "KB0022374",
	    "KB0028566", "KB0025082", "KB0016532", "KB0016430", "KB0023609",
	    "KB0026078", "KB0021768", "KB0027215", "KB0013957", "KB0026085",
	    "KB0027216", "KB0028007", "KB0026871", "KB0026486", "KB0027200",
	    "KB0026081", "KB0014050", "KB0029602", "KB0021486", "KB0026276",
	    "KB0016617", "KB0025081", "KB0011281", "KB0028006", "KB0014077",
	    "KB0028640", "KB0016531", "KB0025923", "KB0010502", "KB0011073",
	    "KB0014134", "KB0023891", "KB0014222", "KB0016995", "KB0016099",
	    "KB0010597", "KB0023604", "KB0029147", "KB0014290", "KB0026089",
	    "KB0017014", "KB0026083", "KB0023617", "KB0016334", "KB0015644",
	    "KB0013105", "KB0027193", "KB0021798", "KB0020628", "KB0029450",
	    "KB0029498", "KB0026811", "KB0027205", "KB0024197", "KB0024792",
	    "KB0027767", "KB0021988", "KB0016925", "KB0013910", "KB0014399",
	    "KB0022796", "KB0021597", "KB0020316", "KB0015973", "KB0013850",
	    "KB0024824", "KB0016992", "KB0021931", "KB0023021", "KB0014055",
	    "KB0016993", "KB0027149", "KB0015840", "KB0027305", "KB0013906",
	    "KB0016977", "KB0014110", "KB0027478", "KB0021767", "KB0027173",
	    "KB0028745", "KB0015903", "KB0015899", "KB0021238"
	]

	old_kb_numbers = [
	    "KB0027217", "KB0026014", "KB0026017", "KB0017014", "KB0026036",
	    "KB0026015", "KB0026033", "KB0028800", "KB0021767", "KB0027173",
	    "KB0022553", "KB0016977", "KB0016993", "KB0016976", "KB0016992",
	    "KB0020015", "KB0027488", "KB0024229", "KB0027505", "KB0014055",
	    "KB0016617", "KB0016506", "KB0014055", "KB0025526", "KB0014110",
	    "KB0014399", "KB0023604", "KB0014273", "KB0026790", "KB0011115",
	    "KB0020316", "KB0025081", "KB0015903", "KB0026668", "KB0026811",
	    "KB0021238", "KB0026689", "KB0024122", "KB0026668", "KB0021597",
	    "KB0020301", "KB0010502", "KB0022137", "KB0028869", "KB0015899",
	    "KB0027578", "KB0023891", "KB0027767", "KB0027505", "KB0025460",
	    "KB0011115", "KB0020316", "KB0025081", "KB0027478", "KB0024226",
	    "KB0026332", "KB0020855", "KB0021920", "KB0022328", "KB0025909",
	    "KB0024792", "KB0024542", "KB0025082", "KB0024843", "KB0024535",
	    "KB0024824", "KB0024197", "KB0024842"
	]

	new_kb_numbers.extend(old_kb_numbers)
	new_kb_numbers = list(set(new_kb_numbers))

	print(len(new_kb_numbers))

	#query_documents = process_documents_list(kb_list)
	query_documents = process_documents_list(["KB0024792"])

	# with open('query_documents.pickle', 'wb') as f:
	# 	pickle.dump(query_documents, f)

	print(f"\nSuccessfully saved {len(query_documents)} query documents")



# Netskope issue synthesis -- process 
