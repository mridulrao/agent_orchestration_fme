import os
import time
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from openai import OpenAI, AzureOpenAI
import numpy as np

import faiss
import pickle
import sentence_transformers

from rag.follow_up_query import QueryAnalyzer
#from follow_up_query import QueryAnalyzer

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KnowledgeQueryEngine")


class KnowledgeQueryEngine:
    """
    Query engine for retrieving knowledge using progressive disclosure.
    
    Workflow:
    1. Take user query
    2. Determine if it's a follow-up or new query
    3. For new queries:
       a. Rewrite/improve query
       b. Convert query to embedding
       c. Retrieve relevant chunks from FAISS index
       d. Rerank chunks by relevance
    4. For follow-ups: 
       a. Use next unused chunk from previous query
    5. Optionally generate answer
    6. Return results with option for more information
    """
    
    def __init__(
        self,
        openai_api_key: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        completion_model: str = "gpt-4o-mini",
        result_count: int = 10,
        query_analyzer_threshold: float = 0.25,
        debug_mode: bool = False,
        use_openai_embeddings: bool = False,
        openai_embedding_model: str = "text-embedding-3-small"
    ):        
        # Store configuration
        self.use_openai_embeddings = use_openai_embeddings
        self.openai_embedding_model = openai_embedding_model
        self.completion_model = completion_model
        self.result_count = result_count
        self.debug_mode = debug_mode
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Load appropriate query documents based on embedding type
        if use_openai_embeddings:
            #query_doc_filepath = os.path.join('..', 'indexing', 'query_documents_openai.pickle')
            #query_idx_filepath = os.path.join('..', 'indexing', 'query_vectors_openai.index')

            # query_doc_filepath = os.path.join('indexing', 'query_documents_openai.pickle')
            # query_idx_filepath = os.path.join('indexing', 'query_vectors_openai.index')

            query_doc_filepath = os.path.join('indexing', 'query_documents_uat.pickle')
            query_idx_filepath = os.path.join('indexing', 'query_vectors_uat.index')

            # query_doc_filepath = os.path.join('..', 'indexing', 'query_documents_uat.pickle')
            # query_idx_filepath = os.path.join('..', 'indexing', 'query_vectors_uat.index')
        else:
            #query_doc_filepath = os.path.join('..', 'indexing', 'query_documents.pickle')
            #query_idx_filepath = os.path.join('..', 'indexing', 'query_vectors.index')

            query_doc_filepath = os.path.join('indexing', 'query_documents.pickle')
            query_idx_filepath = os.path.join('indexing', 'query_vectors.index')
        
        # Load pre-indexed query documents
        with open(query_doc_filepath, 'rb') as f:
            self.query_documents = pickle.load(f)
        
        # Prepare queries and vectors
        self.queries = list(self.query_documents.keys())
        self.vectors = np.array([self.query_documents[query]['query_vector'] for query in self.queries], dtype=np.float32)
        
        # Initialize embedding model based on flag
        if use_openai_embeddings:
            logger.info(f"Using OpenAI embeddings with model: {openai_embedding_model}")
            self.embedding_model = None  # No sentence transformer needed
        else:
            # Load sentence transformer model
            #embedding_model_path = os.path.join('models', 'all-MiniLM-L6-v2')
            embedding_model_path = os.path.join('rag', 'models', embedding_model)
            self.embedding_model = sentence_transformers.SentenceTransformer(embedding_model_path)
            logger.info(f"Using SentenceTransformer model: {embedding_model}")
        
        # Load FAISS index
        self.index = faiss.read_index(query_idx_filepath)
        
        # Initialize the query analyzer
        self.query_analyzer = QueryAnalyzer(
            threshold=query_analyzer_threshold,
            embedding_model=self.embedding_model,
            use_openai_embeddings=use_openai_embeddings,
            openai_client=self.openai_client if use_openai_embeddings else None,
            openai_embedding_model=openai_embedding_model if use_openai_embeddings else None
        )
        
        # Conversation context storage
        self.conversation_context = {
            "current_query": "",
            "unused_chunks": [],
            "query_history": []
        }

    def embed_query_openai(self, query: str) -> np.ndarray:
        """
        Create an embedding vector for the query using OpenAI.
        
        Args:
            query: The query to embed
            
        Returns:
            np.ndarray: The embedding vector normalized for FAISS
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.openai_embedding_model,
                input=query
            )
            
            embedding = response.data[0].embedding
            
            # Convert to numpy array and normalize for FAISS
            user_query_vector = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(user_query_vector)
            
            if self.debug_mode:
                logger.info(f"Created OpenAI embedding with shape: {user_query_vector.shape}")
            
            return user_query_vector
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {str(e)}")
            # Return zero vector with OpenAI embedding dimension (1536 for text-embedding-3-small)
            embedding_dim = 1536 if "small" in self.openai_embedding_model else 3072
            return np.zeros((1, embedding_dim), dtype=np.float32)

    def embed_query_sentence_transformer(self, query: str) -> np.ndarray:
        """
        Create an embedding vector for the query using SentenceTransformer.
        
        Args:
            query: The query to embed
            
        Returns:
            np.ndarray: The embedding vector normalized for FAISS
        """
        try:
            user_query_vector = self.embedding_model.encode([query])[0]
            user_query_vector = np.array([user_query_vector], dtype=np.float32)
            faiss.normalize_L2(user_query_vector)
            
            if self.debug_mode:
                logger.info(f"Created SentenceTransformer embedding with shape: {user_query_vector.shape}")
            
            return user_query_vector
            
        except Exception as e:
            logger.error(f"SentenceTransformer embedding failed: {str(e)}")
            return np.zeros((1, self.embedding_model.get_sentence_embedding_dimension()), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Create an embedding vector for the query using the configured method.
        
        Args:
            query: The query to embed
            
        Returns:
            np.ndarray: The embedding vector normalized for FAISS
        """
        if self.use_openai_embeddings:
            return self.embed_query_openai(query)
        else:
            return self.embed_query_sentence_transformer(query)

    def get_embedding_info(self) -> dict:
        """
        Get information about the current embedding configuration.
        
        Returns:
            dict: Information about embedding setup
        """
        if self.use_openai_embeddings:
            return {
                "type": "openai",
                "model": self.openai_embedding_model,
                "dimension": 1536 if "small" in self.openai_embedding_model else 3072,
                "query_documents_file": "query_documents_openai.pickle",
                "index_file": "query_vectors_openai.index"
            }
        else:
            return {
                "type": "sentence_transformer", 
                "model": self.embedding_model.model_name if self.embedding_model else "unknown",
                "dimension": self.embedding_model.get_sentence_embedding_dimension() if self.embedding_model else "unknown",
                "query_documents_file": "query_documents.pickle",
                "index_file": "query_vectors.index"
            }

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite the query to improve search results, correcting for STT transcription errors.
        
        Args:
            query: The original user query
            entity_mappings: Dictionary mapping possible misspellings to correct entity names
                
        Returns:
            str: The rewritten query
        """

        entity_mappings = ['TruU', 'Outlook', 'Zoom Rooms', 'Microsoft Outlook', 'BYOD Enrollment', 'OneDrive', 
        'GADI', 'Zoom Phone', 'SAP', 'Mobile Enrollment', 'MacBook', 'iOS', 'Research Applications', 
        'Secure Access Enrollment', 'Netskope', 'Standard & Non-Standard Software Installation', 
        'Standard & Non-Standard', 'Known Folder Move', 'Gilead', 'Re-Authentication', 'Windows', 
        'KB0014110', 'KB0015899', 'Schedueling', 'GPAS', 'Intune', 'Software Center', 'Zoom', 'IDM', 
        'ESD Expert Group', 'ESD', 'MAC', 'Teams', 'Private Access', 'ESD Agent', 'Shared mailbox', 'Android', 'NPA', 'BYOD']
        
        try:
            # Use GPT-4o to improve the query for vector search with entity correction
            system_prompt = '''
            [Identity]: 
                - You are an AI assistant that analyzes user requests and rewrites the query.

            [Instructions]:
            1. Consider the latest user message.
               - Identify the core request or issue in the user's latest message.
               - Correct potential speech-to-text (STT) transcription errors in user messages.

            2. Entity correction:
               - You will receive a JSON mapping of commonly mistranscribed entities.
               - The format is {"possible_misspelling": "correct_entity_name"}.
               - Identify and replace any misspelled entities in the query with their correct forms.
               - For example, replace "true view" with "TruU" if that mapping exists.

            3. Matching criteria:
               - Match based on semantic meaning and user intent, not just exact wording.
               - Be more lenient with word choice and phrasing due to potential STT errors.
               - A query matches if it would satisfy the user's current request, accounting for possible transcription errors.
               - Consider technical context and domain-specific terminology, allowing for slight variations.

            4. **DO NOT:**
               - Create new queries not related to the user's request.
               - Return partial matches or intermediate confidence values.
               - Include any text outside the JSON format in your response.

            OUTPUT FORMAT:
            Return a JSON format - 
            {
                "rewritten_query": "corrected query with proper entity names",
                "corrections": [{"original": "misspelled entity", "corrected": "proper entity"}]
            }
            '''
            
            # Convert entity mappings to a string representation for the prompt
            entity_mappings_str = json.dumps(entity_mappings)
            
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original query: {query}\n\nEntity mappings:\n{entity_mappings_str}\n\nPlease rewrite this query to optimize for semantic search, correcting any entity misspellings."}
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            rewritten_query = result.get('rewritten_query', query)
            corrections = result.get('corrections', [])
            
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Query rewrite failed: {str(e)}")
            return query
    
    # def embed_query(self, query: str) -> np.ndarray:
    #     """
    #     Create an embedding vector for the query.
        
    #     Args:
    #         query: The query to embed
            
    #     Returns:
    #         np.ndarray: The embedding vector
    #     """
        
    #     try:
    #         user_query_vector = self.embedding_model.encode([query])[0]
    #         user_query_vector = np.array([user_query_vector], dtype=np.float32)
    #         faiss.normalize_L2(user_query_vector)
            
    #         return user_query_vector
            
    #     except Exception as e:
    #         logger.error(f"Embedding failed: {str(e)}")
    #         return np.zeros((1, self.embedding_model.get_sentence_embedding_dimension()), dtype=np.float32)
    
    def fetch_relevant_contexts(self, query_embedding: np.ndarray, top_k: int = None) -> List[Dict]:
        """
        Fetch the most relevant context chunks using FAISS.
        
        Args:
            query_embedding: The query embedding vector
            top_k: Number of results to retrieve
            
        Returns:
            List[Dict]: The relevant context chunks
        """
        
        if top_k is None:
            top_k = self.result_count
        
        try:
            # Search using FAISS
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            contexts = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.queries):
                    original_query = self.queries[idx]
                    doc_data = self.query_documents[original_query]
                    
                    contexts.append({
                        'query': original_query,
                        'query_answer': doc_data['query_answer'],
                        'document_title': doc_data['document_title'],
                        'document_id': doc_data['document_idc'],
                        'entities': doc_data['meta_data']['entities'] if 'entities' in doc_data['meta_data'] else [],
                        'similarity_score': float(scores[0][i]),
                        'normalized_score': float(1.0 + scores[0][i]) if scores[0][i] < 0 else float(scores[0][i])
                    })
            
            return contexts
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []
    
    def rerank_contexts(self, query: str, contexts: List[Dict]) -> List[Dict]:
        """
        Rerank contexts using a more sophisticated relevance model.
        
        Args:
            query: The original query
            contexts: The contexts to rerank
            
        Returns:
            List[Dict]: The reranked contexts, or empty list if none are relevant
        """
        if not contexts:
            return []
            
        
        try:
            # Use GPT-4o to rerank by relevance
            context_texts = []
            for i, ctx in enumerate(contexts):
                # Create a context entry with identifier
                context_texts.append(f"[{i}] Query: {ctx['query']}\nAnswer: {ctx['query_answer'][:300]}...")
            
            all_contexts = "\n\n".join(context_texts)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are a relevance ranking expert. Rank the provided knowledge base 
                    chunks based on their relevance to the user's query.
                    If none of the contexts are relevant to the query, return an empty array: []
                    Don't use information outside the context. Be concise. No markdown.
                    
                    You will receive a query and a list of context chunks. Your task is to:
                    1. Evaluate each chunk's relevance to the query
                    2. If any chunks are relevant, return a comma-separated list of the chunk IDs in descending order of relevance
                    3. If NO chunks are relevant to the query, return only "[]" (empty array)
                    4. Do not explain your reasoning
                    
                    Example output format: 4,0,2,1,3
                    This means chunk 4 is most relevant, followed by 0, then 2, etc.
                    
                    If none are relevant, simply respond with:
                    []
                    """},
                    {"role": "user", "content": f"Query: {query}\n\nContext chunks:\n{all_contexts}\n\nRank these chunks by relevance (comma-separated IDs, most relevant first) or return [] if none are relevant:"}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            ranking_result = response.choices[0].message.content.strip()
            
            # Parse the reranking result
            try:
                # Check if result indicates no relevant context
                if ranking_result == "[]":
                    logger.info(f"No relevant contexts found for query: '{query}'")
                    return []
                    
                # Handle possible formatting variations
                ranking_result = ranking_result.replace(" ", "")
                if ranking_result.startswith("[") and ranking_result.endswith("]"):
                    ranking_result = ranking_result[1:-1]
                    
                # If empty array after cleaning, return empty list
                if not ranking_result:
                    logger.info(f"No relevant contexts found for query: '{query}'")
                    return []
                    
                rank_order = [int(idx) for idx in ranking_result.split(",")]
                
                # Create new list with reranked contexts
                reranked = []
                used_indices = set()
                
                # Add contexts in the reranked order
                for idx in rank_order:
                    if 0 <= idx < len(contexts) and idx not in used_indices:
                        reranked.append(contexts[idx])
                        used_indices.add(idx)
                
                # Add any contexts that weren't in the ranking
                for idx, ctx in enumerate(contexts):
                    if idx not in used_indices:
                        reranked.append(ctx)
                
                return reranked
                
            except Exception as e:
                logger.error(f"Failed to parse reranking result '{ranking_result}': {str(e)}")
                return contexts
                
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            return contexts
    
    def format_context_for_llm(self, contexts: List[Dict]) -> List[Dict]:
        """
        Format the contexts into a list of processed chunks ready for progressive disclosure.
        
        Args:
            contexts: The contexts to format
            
        Returns:
            List[Dict]: The processed chunks with pre-generated answers
        """
        processed_chunks = []
        
        # Process each context into a ready-to-use chunk
        for i, ctx in enumerate(contexts):
            # Create a chunk with all necessary information
            processed_chunk = {
                "chunk_id": i + 1,  # 1-indexed for readability
                "query": ctx["query"],
                "answer": ctx["query_answer"],
                "document_title": ctx["document_title"],
                "document_id": ctx["document_id"],
                "relevance": ctx["normalized_score"],
                "used": False  # Track if this chunk has been used
            }
            
            processed_chunks.append(processed_chunk)
        
        return processed_chunks
    
    def generate_answer_from_chunk(self, query: str, chunk: Dict) -> str:
        """
        Generate an answer from a single context chunk.
        
        Args:
            query: The user's query
            chunk: The single context chunk to use
            
        Returns:
            str: The generated response
        """
        
        # Create a simplified prompt focused on this single chunk
        system_prompt = '''
        You are a knowledge assistant that answers based on provided context. 
        Don't use information outside the context. If context is irrelevant, return "No relevant context found" 
        Be concise. No markdown.
        '''
        
        chunk_context = f"""ORIGINAL QUERY: {chunk['query']}
        ANSWER:
        {chunk['answer']}
        """
        
        user_prompt = f"""
        User Query: {query}
        Context: {chunk_context}
        
        Provide a helpful answer to the user query using only the above context.
        If the context seems irrelevant to the query, say so politely without making up information.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.completion_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            
            return result
            
        except Exception as e:
            logger.error(f"Chunk response generation failed: {str(e)}")
            return "I couldn't generate an answer from this knowledge source."
    
    def create_query_context(self, query: str, query_history: List[str]) -> Dict[str, Any]:
        """
        Create context for the QueryAnalyzer.
        
        Args:
            query: The current user query
            query_history: List of previous queries
            
        Returns:
            Dict: Context information for the analyzer
        """
        return {
            "current_query": query,
            "query_history": query_history
        }


    def progressive_query(self, query: str, generate_answer: bool = True, include_debug_info: bool = False) -> Dict[str, Any]:
        """
        Process a new query through the pipeline.
        
        Args:
            query: The user's query
            generate_answer: Whether to generate an answer or just return ranked chunks
            include_debug_info: Whether to include timing and debug info in response
            
        Returns:
            Dict: The response with timing data if requested
        """
        
        # # Step 1: Rewrite query
        # step_start = time.time()
        # rewritten_query = self.rewrite_query(query)
        # steps_timing["rewrite_query"] = time.time() - step_start

        # already rewriting the query when analyzing follow up
        rewritten_query = query
        
        # Step 2: Embed query
        query_embedding = self.embed_query(rewritten_query)
        
        # Step 3: Fetch relevant contexts
        contexts = self.fetch_relevant_contexts(query_embedding, top_k=self.result_count)
        
        # Step 4: Rerank contexts
        contexts = self.rerank_contexts(query, contexts)
        
        # Prepare result
        result = {
            "query": query,
            "query_type": "new"  # Mark this as a new query
        }
        
        # Check if no relevant contexts were found
        if not contexts:
            logger.info("No relevant information found for query")
            result["response"] = "I don't have information about that topic in my knowledge base." if generate_answer else None
            result["sources"] = []
            result["has_more_info"] = False
            result["unused_chunks"] = []
            
            return result
        
        # Step 5: Format contexts for progressive disclosure (only if we have contexts)
        processed_chunks = self.format_context_for_llm(contexts[:self.result_count])
        
        # Step 6: Generate response if requested
        if generate_answer and processed_chunks:
            top_chunk = processed_chunks[0]
            top_chunk["used"] = True
            response = self.generate_answer_from_chunk(query, top_chunk)
            
            # Add response and source to result
            result["response"] = response
            result["sources"] = [{
                "id": top_chunk["chunk_id"],
                "query": top_chunk["query"],
                "document_title": top_chunk["document_title"],
                "document_id": top_chunk["document_id"]
            }]
            
            # Store the remaining chunks for later use
            unused_chunks = [chunk for chunk in processed_chunks if not chunk["used"]]
            result["has_more_info"] = len(unused_chunks) > 0
            result["unused_chunks"] = unused_chunks
        else:
            # Only return the top 1 chunks when not generating an answer
            displayed_chunks = processed_chunks[:1]
            result["processed_chunks"] = displayed_chunks
            
            # Store all chunks as unused for future follow-ups, but mark the first 1 as used
            for i, chunk in enumerate(processed_chunks):
                chunk["used"] = i < 1  # Mark first 2 chunks as used
            
            unused_chunks = [chunk for chunk in processed_chunks if not chunk["used"]]
            result["has_more_info"] = len(unused_chunks) > 0
            result["unused_chunks"] = unused_chunks
        
        return result
    
    def get_more_info(self, query: str, unused_chunks: List[Dict], generate_answer: bool = True) -> Dict[str, Any]:
        """
        Get additional information from unused chunks, with optional answer generation.
        
        Args:
            query: The original query
            unused_chunks: List of unused chunks from previous query
            generate_answer: Whether to generate an answer or just return the chunk
            
        Returns:
            Dict: The response with the next best chunk
        """
        
        if not unused_chunks:
            return {
                "query": query,
                "response": "No additional information available." if generate_answer else None,
                "sources": [],
                "has_more_info": False,
                "query_type": "follow_up"
            }
        
        # Get the highest-relevance unused chunk
        next_chunk = unused_chunks[0]
        next_chunk["used"] = True
        
        # Prepare result
        result = {
            "query": query,
            "chunk": next_chunk,  # Keep this for backward compatibility
            "processed_chunks": [next_chunk],  # Add this line to include processed_chunks
            "sources": [{
                "id": next_chunk["chunk_id"],
                "query": next_chunk["query"],
                "document_title": next_chunk["document_title"],
                "document_id": next_chunk["document_id"]
            }],
            "has_more_info": len(unused_chunks[1:]) > 0,
            "unused_chunks": unused_chunks[1:],
            "query_type": "follow_up"
        }
        
        # Generate response if requested
        if generate_answer:
            response = self.generate_answer_from_chunk(query, next_chunk)
            result["response"] = response
        
        return result
    
    def handle_query_with_analyzer(self, query: str, generate_answer: bool = True, include_debug_info: bool = False) -> Dict[str, Any]:
        """
        Process a query using the QueryAnalyzer to determine if it's a new query or follow-up.
        
        Args:
            query: The user's query
            generate_answer: Whether to generate an answer or just return ranked chunks
            include_debug_info: Whether to include timing and debug info in response
            
        Returns:
            Dict: The response with query type determination
        """
        # Extract info from current context
        current_query = self.conversation_context.get("current_query", "")
        unused_chunks = self.conversation_context.get("unused_chunks", [])
        query_history = self.conversation_context.get("query_history", [])

        query = self.rewrite_query(query)
        
        # Add to query history
        query_history.append(query)
        
        self.conversation_context["current_query"] = query
        
        # Analyze the query to determine if it's a follow-up
        analysis = self.query_analyzer.analyze_query(query, verbose=self.debug_mode)
        
        if analysis['type'] == 'follow_up':
            
            result = self.get_more_info(current_query, unused_chunks, generate_answer)
            result["analyzer_result"] = analysis
            
            # Update conversation context
            self.conversation_context["unused_chunks"] = result.get("unused_chunks", [])
            self.conversation_context["query_history"] = query_history
            
            return result
        else:
            
            result = self.progressive_query(query, generate_answer, include_debug_info)
            result["analyzer_result"] = analysis
            
            # Update conversation context
            self.conversation_context["current_query"] = query
            self.conversation_context["unused_chunks"] = result.get("unused_chunks", [])
            self.conversation_context["query_history"] = query_history
            
            return result
    
    def query(self, query: str, generate_answer: bool = False, include_debug_info: bool = False) -> Dict[str, Any]:
        """
        Main query method - now uses query analyzer for intelligent progressive disclosure
        
        Args:
            query: The user's query
            generate_answer: Whether to generate an answer or just return ranked chunks
            include_debug_info: Whether to include timing and debug info
            
        Returns:
            Dict: The response with progressive disclosure
        """

        rag_responce = {
            'unprocessed_query': query,
            'processed_query': None,
            'answer' : None,
            'refefrenced_doc_ids': None,
            'refefrenced_doc_title': None
        }


        result = self.handle_query_with_analyzer(query, generate_answer, include_debug_info)

        try:
            kb_answer = ''
            refefrenced_doc_ids = []
            refefrenced_doc_title = []

            for chunk in result['processed_chunks']:
                kb_answer += chunk['answer'] + '\n'
                refefrenced_doc_ids.append(chunk['document_id'])
                refefrenced_doc_title.append(chunk['document_title'])

            rag_responce['processed_query'] = result['query']
            rag_responce['answer'] = kb_answer
            rag_responce['refefrenced_doc_ids'] = refefrenced_doc_ids
            rag_responce['refefrenced_doc_title'] = refefrenced_doc_title

            return rag_responce

        except Exception as e:
            print(f"Error in processing the query: {e}")
            return None


## Example usage
# if __name__ == "__main__":
#     # Initialize the query engine
#     openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    
#     query_engine = KnowledgeQueryEngine(
#         openai_api_key=openai_api_key,
#         use_openai_embeddings=True,
#         openai_embedding_model="text-embedding-3-small"
#     )
    
#     # State tracking
#     generate_answer = False
    
#     while True:
#         query = input(f"\nEnter your query (or 'raw', 'exit') [Answer generation: {'ON' if generate_answer else 'OFF'}]: ")
        
#         if query.lower() in ("exit", "quit"):
#             break
            
#         if query.lower() == "raw":
#             generate_answer = not generate_answer
#             print(f"\nAnswer generation turned {'ON' if generate_answer else 'OFF'}")
#             continue
        
#         print("\nProcessing query...")
        
#         # Process the query - the system automatically determines if it's a follow-up
#         result = query_engine.query(query, generate_answer, include_debug_info=False)
        
#         if result:
#             print(result)
#         else:
#             print("No information found")
        


