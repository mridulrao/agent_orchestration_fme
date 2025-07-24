import os
from pathlib import Path
import time
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from lancedb import connect
import numpy as np
from bedrock_runtime_client import BedrockClientSingleton  # Import your singleton

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("KnowledgeQueryEngine")

class KnowledgeQueryEngine:
    """
    Query engine for retrieving knowledge from LanceDB and generating responses.

    Workflow:
    1. Take user query
    2. Rewrite/improve query
    3. Convert query to embedding
    4. Retrieve relevant chunks from LanceDB
    5. Rerank chunks by relevance
    6. Send to LLM with context for response generation
    7. Return response with citations
    """

    def __init__(
        self,
        lancedb_uri: str,
        lancedb_table_name: str = "servicenow_knowledge",
        embedding_model: str = "amazon.titan-embed-text-v2:0",
        completion_model: str = "anthropic.claude-4-sonnet",  # Updated for Claude
        result_count: int = 5,
        reranking_enabled: bool = True,
        debug_mode: bool = False,
    ):
        """
        Initialize the knowledge query engine.

        Args:
            lancedb_uri: URI for LanceDB database
            lancedb_table_name: Name of the LanceDB table
            embedding_model: Name of the embedding model to use
            completion_model: Name of the completion model to use (Claude on Bedrock)
            result_count: Number of results to return from vector search
            reranking_enabled: Whether to rerank results
            debug_mode: Whether to log detailed debugging information
        """
        start_time = time.time()

        # Set up connections
        self.lancedb_uri = lancedb_uri
        self.db = connect(lancedb_uri)
        self.table_name = lancedb_table_name

        # Check if table exists
        if self.table_name not in self.db.table_names():
            print(f"LanceDB URI: {lancedb_uri}")
            raise ValueError(f"Table '{self.table_name}' not found in LanceDB")

        self.table = self.db.open_table(self.table_name)

        # Store parameters
        self.embedding_model = embedding_model
        self.completion_model = completion_model
        self.result_count = result_count
        self.reranking_enabled = reranking_enabled
        self.debug_mode = debug_mode

        # Get the Bedrock client from singleton
        self.bedrock_singleton = BedrockClientSingleton()
        # Test connection
        if self.bedrock_singleton.test_connection():
            logger.info("Bedrock client ready for use")

        self.bedrock_client = self.bedrock_singleton.get_client()

        # Log initialization time
        init_time = time.time() - start_time
        logger.info(f"Initialized KnowledgeQueryEngine in {init_time:.2f} seconds")
        logger.info(f"Using LanceDB table: {self.table_name}")
        logger.info(f"Using embedding model: {self.embedding_model}")
        logger.info(f"Using completion model: {self.completion_model}")

        # Log table info
        try:
            df = self.table.to_pandas()
            article_count = len(df["sys_id"].unique())
            chunk_count = len(df)
            logger.info(
                f"Found {article_count} articles with {chunk_count} chunks in the database"
            )
        except Exception as e:
            logger.error(f"Error getting table stats: {str(e)}")

    def _call_claude(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.1, 
        max_tokens: int = 300
    ) -> str:
        """
        Call Claude using Bedrock client singleton.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Claude's response
        """
        try:
            # Convert messages to Claude format (handle system messages)
            claude_messages = []
            system_content = ""
            
            for message in messages:
                if message["role"] == "system":
                    system_content = message["content"]
                elif message["role"] in ["user", "assistant"]:
                    claude_messages.append({
                        "role": message["role"],
                        "content": message["content"]
                    })
            
            # If we have a system message, prepend it to the first user message
            if system_content and claude_messages and claude_messages[0]["role"] == "user":
                claude_messages[0]["content"] = f"{system_content}\n\nUser: {claude_messages[0]['content']}"
            
            # Prepare request body
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": claude_messages
            }
            
            # Call Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.completion_model,
                body=json.dumps(body),
                contentType='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            return response_body.get('content', [{}])[0].get('text', '')
            
        except Exception as e:
            logger.error(f"Claude API call failed: {str(e)}")
            raise

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite the query to improve search results using Claude.

        Args:
            query: The original user query

        Returns:
            str: The rewritten query
        """
        start_time = time.time()

        try:
            response = self._call_claude(
                messages=[
                    {
                        "role": "system",
                        "content": """You are a query optimization assistant. Your task is to rewrite 
                    the user's query to be more effective for semantic vector search in a knowledge base.
                    
                    Guidelines:
                    - Expand abbreviations and technical terms
                    - Add relevant synonyms and related concepts
                    - Remove unnecessary words like "how to" or "what is" if they don't add meaning
                    - Focus on key technical terms and concepts
                    - Maintain the core intent of the original query
                    - The output should be just the rewritten query, nothing else
                    
                    Your response will be used directly as input to a vector search.""",
                    },
                    {
                        "role": "user",
                        "content": f"Original query: {query}\n\nPlease rewrite this query to optimize for semantic search.",
                    },
                ],
                temperature=0.1,
                max_tokens=300,
            )

            rewritten_query = response.strip()

            # Log the rewrite
            rewrite_time = time.time() - start_time
            logger.info(f"Rewrote query in {rewrite_time:.2f} seconds")
            logger.info(f"Original query: '{query}'")
            logger.info(f"Rewritten query: '{rewritten_query}'")

            return rewritten_query

        except Exception as e:
            logger.error(f"Query rewrite failed: {str(e)}")
            logger.info("Using original query")
            return query

    def embed_query(self, query: str) -> List[float]:
        """
        Create an embedding vector for the query using Bedrock singleton.

        Args:
            query: The query to embed

        Returns:
            List[float]: The embedding vector
        """
        start_time = time.time()

        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.embedding_model,
                body=json.dumps(
                    {
                        "inputText": query,
                        "dimensions": 1024,
                        "normalize": True,
                    }
                ),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response.get("body").read())
            embeddings = response_body["embedding"]

            # Log embedding time
            embed_time = time.time() - start_time
            logger.info(f"Created embedding in {embed_time:.2f} seconds")

            return embeddings

        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 1024  # Match dimensionality of amazon.titan-embed-text-v2:0

    def fetch_relevant_contexts(
        self, query_embedding: List[float], top_k: int = 10
    ) -> List[Dict]:
        """
        Fetch the most relevant context chunks from LanceDB.

        Args:
            query_embedding: The query embedding vector
            top_k: Number of results to retrieve

        Returns:
            List[Dict]: The relevant context chunks
        """
        start_time = time.time()

        try:
            # Convert query embedding to proper format for LanceDB
            query_vector = np.array(query_embedding, dtype=np.float32)

            # Perform the vector search
            results = self.table.search(query_vector).limit(top_k).to_pandas()

            # Convert to list of dictionaries
            contexts = []
            for _, row in results.iterrows():
                contexts.append(
                    {
                        "sys_id": row.get("sys_id", ""),
                        "title": row.get("title", ""),
                        "chunk_index": row.get("chunk_index", 0),
                        "text_chunk": row.get("text_chunk", ""),
                        "score": float(
                            row.get("_distance", 0.0)
                        ),  # LanceDB returns distance
                        "ai_metadata": row.get("ai_metadata", {}),
                    }
                )

            # Add numeric scores (1.0 is closest, 0.0 is furthest in our normalization)
            if contexts:
                max_score = max(context["score"] for context in contexts)
                min_score = min(context["score"] for context in contexts)
                score_range = max_score - min_score if max_score > min_score else 1.0

                for context in contexts:
                    # Convert distance to normalized similarity score (1.0 is best)
                    context["normalized_score"] = 1.0 - (
                        (context["score"] - min_score) / score_range
                    )

            # Log the search time
            search_time = time.time() - start_time
            context_count = len(contexts)
            logger.info(
                f"Found {context_count} relevant chunks in {search_time:.2f} seconds"
            )

            if self.debug_mode:
                # Log the top 3 results for debugging
                for i, ctx in enumerate(contexts[:3]):
                    logger.debug(f"Context {i+1}:")
                    logger.debug(f"  Title: {ctx['title']}")
                    logger.debug(f"  Score: {ctx['normalized_score']:.4f}")
                    logger.debug(f"  Text: {ctx['text_chunk'][:150]}...")

            return contexts

        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    def rerank_contexts(self, query: str, contexts: List[Dict]) -> List[Dict]:
        """
        Rerank contexts using Claude for more sophisticated relevance scoring.

        Args:
            query: The original query
            contexts: The contexts to rerank

        Returns:
            List[Dict]: The reranked contexts
        """
        if not self.reranking_enabled or not contexts:
            return contexts

        start_time = time.time()

        try:
            # Create context texts for reranking
            context_texts = []
            for i, ctx in enumerate(contexts):
                context_texts.append(
                    f"[{i}] Title: {ctx['title']}\nContent: {ctx['text_chunk'][:500]}..."
                )

            all_contexts = "\n\n".join(context_texts)

            response = self._call_claude(
                messages=[
                    {
                        "role": "system",
                        "content": """You are a relevance ranking assistant. Given a query and multiple context chunks, rank them by relevance to the query.
                        
                        Guidelines:
                        - Consider semantic similarity and topical relevance
                        - Prioritize chunks that directly answer the query
                        - Consider completeness and specificity of information
                        - Return only the ranking as comma-separated numbers (e.g., "2,0,1,3")
                        - Most relevant first, least relevant last""",
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nContext chunks:\n{all_contexts}\n\nRank these chunks by relevance (comma-separated IDs, most relevant first):",
                    },
                ],
                temperature=0.1,
                max_tokens=50,
            )

            ranking_result = response.strip()

            # Parse the reranking result
            try:
                # Handle possible formatting variations
                ranking_result = ranking_result.replace(" ", "")
                if ranking_result.startswith("[") and ranking_result.endswith("]"):
                    ranking_result = ranking_result[1:-1]

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

                rerank_time = time.time() - start_time
                logger.info(f"Reranked contexts in {rerank_time:.2f} seconds")
                logger.info(f"Reranking order: {ranking_result}")

                return reranked

            except Exception as e:
                logger.error(
                    f"Failed to parse reranking result '{ranking_result}': {str(e)}"
                )
                return contexts

        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            return contexts

    def format_context_for_llm(
        self, contexts: List[Dict], max_tokens: int = 6000
    ) -> Tuple[str, List[Dict]]:
        """
        Format the contexts into a string for the LLM prompt.
        Ensures the context stays within token limits.

        Args:
            contexts: The contexts to format
            max_tokens: Approximate maximum number of tokens to include

        Returns:
            Tuple[str, List[Dict]]: The formatted context and the list of used sources for citations
        """
        formatted_chunks = []
        used_sources = []
        estimated_tokens = 0.0

        # Estimate tokens (roughly 4 characters per token)
        for i, ctx in enumerate(contexts):
            # Create a citation ID for this chunk
            citation_id = f"[{i+1}]"

            chunk = f"""DOCUMENT {citation_id}: {ctx['title']}
CONTENT:
{ctx['text_chunk']}
RELEVANCE: {ctx['normalized_score']:.2f}
---
"""
            # Rough token estimation
            token_estimate = len(chunk) / 4

            if estimated_tokens + token_estimate > max_tokens:
                # Stop adding chunks if we're near the limit
                break

            formatted_chunks.append(chunk)
            estimated_tokens += token_estimate

            # Track this source for citations
            used_sources.append(
                {
                    "citation_id": i + 1,  # 1-indexed for readability
                    "sys_id": ctx["sys_id"],
                    "title": ctx["title"],
                    "relevance": ctx["normalized_score"],
                }
            )

        if not formatted_chunks:
            return "No relevant knowledge found.", []

        return "\n".join(formatted_chunks), used_sources

    def format_citations(self, sources: List[Dict]) -> List[Dict]:
        """
        Format the citations from the used sources.

        Args:
            sources: List of source documents used

        Returns:
            List[Dict]: Formatted citation objects
        """
        if not sources:
            return []

        # Remove duplicates (same article, different chunks)
        unique_sources: Dict[str, Dict] = {}
        for source in sources:
            sys_id = source["sys_id"]
            if (
                sys_id not in unique_sources
                or source["relevance"] > unique_sources[sys_id]["relevance"]
            ):
                unique_sources[sys_id] = source

        return list(unique_sources.values())

    def generate_response(
        self, query: str, formatted_context: str, sources: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        Generate a response using Claude based on the query and context.

        Args:
            query: The user's query
            formatted_context: The formatted context from knowledge base
            sources: The list of sources used for context

        Returns:
            Tuple[str, List[Dict]]: The generated response and formatted citations
        """
        start_time = time.time()

        system_prompt = """You are a helpful knowledge assistant. Your task is to provide accurate, helpful answers based on the provided context from a knowledge base.

Guidelines:
- Answer based on the provided context documents
- If the context doesn't contain enough information, say so clearly
- Include relevant citations using the document numbers [1], [2], etc.
- Be concise but comprehensive
- If multiple documents provide similar information, synthesize them
- If there are contradictions in the documents, mention them
- Maintain a professional, helpful tone"""

        user_prompt = f"""Context from knowledge base:
{formatted_context}

User question: {query}

Please provide a helpful answer based on the context above. Include citations to relevant documents using [1], [2], etc."""

        try:
            result = self._call_claude(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            # Format citations for the response
            citations = self.format_citations(sources)

            # Log generation time
            generation_time = time.time() - start_time
            logger.info(
                f"Generated response in {generation_time:.2f} seconds ({len(result)} chars)"
            )
            logger.info(
                f"Using {len(sources)} source chunks from {len(set(s['sys_id'] for s in sources))} unique articles"
            )

            return result, citations

        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return (
                "I'm sorry, I encountered an error while generating a response.",
                [],
            )

    def query(self, query: str, include_debug_info: bool = False) -> Dict[str, Any]:
        """
        Process a complete query through the pipeline.

        Args:
            query: The user's query
            include_debug_info: Whether to include timing and debug info in response

        Returns:
            Dict: The response with timing data if requested
        """
        start_time = time.time()
        steps_timing = {}

        # Step 1: Rewrite query
        step_start = time.time()
        rewritten_query = self.rewrite_query(query)
        steps_timing["rewrite_query"] = time.time() - step_start

        # Step 2: Embed query
        step_start = time.time()
        query_embedding = self.embed_query(rewritten_query)
        steps_timing["embed_query"] = time.time() - step_start

        # Step 3: Fetch relevant contexts
        step_start = time.time()
        contexts = self.fetch_relevant_contexts(
            query_embedding, top_k=self.result_count * 2
        )
        steps_timing["fetch_contexts"] = time.time() - step_start

        # Step 4: Rerank contexts if enabled
        step_start = time.time()
        if self.reranking_enabled:
            contexts = self.rerank_contexts(query, contexts)
        steps_timing["rerank_contexts"] = time.time() - step_start

        # Step 5: Format contexts for LLM
        step_start = time.time()
        formatted_context, used_sources = self.format_context_for_llm(
            contexts[: self.result_count]
        )
        steps_timing["format_context"] = time.time() - step_start

        # Step 6: Generate response
        step_start = time.time()
        response, citations = self.generate_response(
            query, formatted_context, used_sources
        )
        steps_timing["generate_response"] = time.time() - step_start

        # Calculate total time
        total_time = time.time() - start_time

        # Log total processing time
        logger.info(f"Total query processing time: {total_time:.2f} seconds")
        logger.info("Query processing complete")

        # Prepare result
        result = {
            "query": query,
            "response": response,
            "sources": [
                {
                    "id": src["citation_id"],
                    "title": src["title"],
                    "sys_id": src["sys_id"],
                }
                for src in citations
            ],
        }

        # Include debug info if requested
        if include_debug_info:
            result["debug_info"] = {
                "rewritten_query": rewritten_query,
                "timings": {"total_seconds": total_time, "steps": steps_timing},
                "context_count": len(contexts),
                "selected_contexts": [
                    {
                        "title": ctx["title"],
                        "score": ctx["normalized_score"],
                        "text_preview": ctx["text_chunk"][:100] + "...",
                    }
                    for ctx in contexts[:3]  # Include just top 3 for brevity
                ],
            }

        return result


# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Query Engine")
    parser.add_argument(
        "--query", type=str, help="Query to run against the knowledge base"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Get environment variables with defaults
    current_dir = Path(__file__).parent  # rag directory
    project_root = current_dir.parent    # project root  
    lancedb_uri = str(project_root / "indexing" / "fme_vector_index")

    #lancedb_uri = os.environ.get("LANCEDB_URI", "./indexing/fme_vector_index")

    # Initialize the query engine (removed openai_api_key parameter)
    query_engine = KnowledgeQueryEngine(
        lancedb_uri=lancedb_uri, 
        debug_mode=args.debug
    )

    # Interactive mode if no query provided
    if not args.query:
        print("Interactive Knowledge Query Mode (type 'exit' to quit)")
        while True:
            query = input("\nEnter your query: ")
            if query.lower() in ("exit", "quit"):
                break

            print("\nProcessing query...")
            result = query_engine.query(query, include_debug_info=args.debug)
            print("\n" + "=" * 80)
            print("RESPONSE:")
            print(result["response"])
            print("\nCITATIONS:")
            for source in result["sources"]:
                print(f"[{source['id']}] {source['title']} (ID: {source['sys_id']})")
            print("=" * 80)

            if args.debug and "debug_info" in result:
                debug = result["debug_info"]
                print("\nDEBUG INFO:")
                print(f"Rewritten query: {debug['rewritten_query']}")
                print(f"Total time: {debug['timings']['total_seconds']:.2f} seconds")
                print("Step times:")
                for step, time_taken in debug["timings"]["steps"].items():
                    print(f"  {step}: {time_taken:.2f}s")
    else:
        # Run the provided query
        result = query_engine.query(args.query, include_debug_info=args.debug)
        print(result["response"])
        print("\nCITATIONS:")
        for source in result["sources"]:
            print(f"[{source['id']}] {source['title']} (ID: {source['sys_id']})")

        if args.debug and "debug_info" in result:
            print("\nDEBUG INFO:")
            print(json.dumps(result["debug_info"], indent=2))