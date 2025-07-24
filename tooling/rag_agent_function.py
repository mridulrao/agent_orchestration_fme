import os
import time
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from openai import OpenAI, AzureOpenAI
from lancedb import connect
import numpy as np
from service_desk.config import init_config
from service_desk.embedding_provider.embedding_factory import EmbeddingProviderFactory
from service_desk.ml.llms.open_ai import chat_completion_request

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("KnowledgeQueryEngine")

cfg = init_config()


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
        openai_api_key: str,
        lancedb_table_name: str = "servicenow_knowledge",
        embedding_model: str = cfg.openai.EMBEDDING_MODEL,
        completion_model: str = cfg.openai.CONVERSATION_MODEL_ADVANCED,
        result_count: int = 5,
        reranking_enabled: bool = True,
        debug_mode: bool = False,
    ):
        """
        Initialize the knowledge query engine.

        Args:
            lancedb_uri: URI for LanceDB database
            openai_api_key: OpenAI API key
            lancedb_table_name: Name of the LanceDB table
            embedding_model: Name of the embedding model to use
            completion_model: Name of the completion model to use
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
            raise ValueError(f"Table '{self.table_name}' not found in LanceDB")

        self.table = self.db.open_table(self.table_name)

        # Store parameters
        self.embedding_model = embedding_model
        self.completion_model = completion_model
        self.result_count = result_count
        self.reranking_enabled = reranking_enabled
        self.debug_mode = debug_mode

        # Initialize embedding provider using the factory
        self.embedding_provider = EmbeddingProviderFactory.create_embedding_provider(
            provider=cfg.embedding.provider,
            model=embedding_model,
            dimensions=cfg.rag.DIMENSIONS,
        )

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

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite the query to improve search results.

        Args:
            query: The original user query

        Returns:
            str: The rewritten query
        """
        start_time = time.time()

        try:
            # Use the configured completion model to improve the query for vector search
            response, _ = chat_completion_request(
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
                model=self.completion_model,
                provider=cfg.provider.llm_provider,
                temperature=0.1,
                max_tokens=300,
            )

            rewritten_query = response.choices[0].message.content.strip()

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
        Create an embedding vector for the query.

        Args:
            query: The query to embed

        Returns:
            List[float]: The embedding vector
        """
        start_time = time.time()

        try:
            embedding = self.embedding_provider.get_text_embedding(query)

            # Log embedding time
            embed_time = time.time() - start_time
            logger.info(f"Created embedding in {embed_time:.2f} seconds")

            return embedding

        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 1536  # Match dimensionality of text-embedding-3-small

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
        Rerank contexts using a more sophisticated relevance model.

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
            # Use the configured completion model to rerank by relevance
            context_texts = []
            for i, ctx in enumerate(contexts):
                # Create a context entry with identifier
                context_texts.append(
                    f"[{i}] Title: {ctx['title']}\nContent: {ctx['text_chunk'][:500]}..."
                )

            all_contexts = "\n\n".join(context_texts)

            response, _ = chat_completion_request(
                messages=[
                    {
                        "role": "system",
                        "content": """You are a relevance ranking expert. Rank the provided knowledge base 
                    chunks based on their relevance to the user's query.
                    
                    You will receive a query and a list of context chunks. Your task is to:
                    1. Evaluate each chunk's relevance to the query
                    2. Return ONLY a comma-separated list of the chunk IDs in descending order of relevance
                    3. Do not explain your reasoning
                    
                    Example output format: 4,0,2,1,3
                    This means chunk 4 is most relevant, followed by 0, then 2, etc.""",
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nContext chunks:\n{all_contexts}\n\nRank these chunks by relevance (comma-separated IDs, most relevant first):",
                    },
                ],
                model=self.completion_model,
                provider=cfg.provider.llm_provider,
                temperature=0.1,
                max_tokens=50,
            )

            ranking_result = response.choices[0].message.content.strip()

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
        Generate a response using the LLM based on the query and context.

        Args:
            query: The user's query
            formatted_context: The formatted context from knowledge base
            sources: The list of sources used for context

        Returns:
            Tuple[str, List[Dict]]: The generated response and formatted citations
        """
        start_time = time.time()

        system_prompt = """You are a knowledgeable assistant that helps users find information from a knowledge base.
Your task is to answer the user's query based on the provided context information.

Guidelines:
- If the context contains the information needed to answer the query, provide a clear and concise response.
- Structure your response logically with paragraphs or bullet points when appropriate.
- Include citation numbers [1], [2], etc. at the end of sentences or paragraphs that reference information from specific sources.
- Use multiple citations [1][2] when information comes from multiple sources.
- If the question cannot be fully answered with the provided context, state what you can answer and what's missing.
- If the context is completely irrelevant to the question, say "I don't have enough information to answer this question."
- Do not reference the context directly in your answer (e.g., don't say "According to the context...")
- Only use information from the provided context. Do not invent information.
- Your answer should be comprehensive but concise.
"""

        user_prompt = f"""Query: {query}

Relevant knowledge base information:
{formatted_context}

Please provide an answer to the query based only on the information provided above.
Include numbered citations (e.g., [1], [2]) at the end of sentences or paragraphs to indicate which documents you're referencing.
Each document in the context is already labeled with a citation number."""

        try:
            response, _ = chat_completion_request(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.completion_model,
                provider=cfg.provider.llm_provider,
                temperature=0.3,
            )

            result = response.choices[0].message.content.strip()

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
    lancedb_uri = os.environ.get("KB_LANCEDB_URI", "./data/vector_index")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is required")
        exit(1)

    # Initialize the query engine
    query_engine = KnowledgeQueryEngine(
        lancedb_uri=lancedb_uri, openai_api_key=openai_api_key, debug_mode=args.debug
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
            print(result["citations"])
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
        print(result["citations"])

        if args.debug and "debug_info" in result:
            print("\nDEBUG INFO:")
            print(json.dumps(result["debug_info"], indent=2))