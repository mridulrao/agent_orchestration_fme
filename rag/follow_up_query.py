import numpy as np
import sentence_transformers
from sklearn.metrics.pairwise import cosine_similarity
import time
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    def __init__(
        self, 
        embedding_model=None, 
        threshold=0.25,
        use_openai_embeddings: bool = False,
        openai_client=None,
        openai_embedding_model: str = "text-embedding-3-small"
    ):
        """Initialize the query analyzer with an embedding model
        
        Args:
            embedding_model: SentenceTransformer model (only used if use_openai_embeddings=False)
            threshold: Similarity threshold for determining topic changes
            use_openai_embeddings: Whether to use OpenAI embeddings
            openai_client: OpenAI client instance (required if use_openai_embeddings=True)
            openai_embedding_model: OpenAI embedding model name
        """
        # Store configuration
        self.use_openai_embeddings = use_openai_embeddings
        self.openai_client = openai_client
        self.openai_embedding_model = openai_embedding_model
        
        # Set up embedding model
        if use_openai_embeddings:
            if openai_client is None:
                raise ValueError("OpenAI client is required when use_openai_embeddings=True")
            self.embedding_model = None
            self.model_available = True
            logger.info(f"QueryAnalyzer initialized with OpenAI embeddings: {openai_embedding_model}")
        else:
            self.embedding_model = embedding_model
            self.model_available = embedding_model is not None
            if self.model_available:
                logger.info(f"QueryAnalyzer initialized with SentenceTransformer: {embedding_model.model_name if hasattr(embedding_model, 'model_name') else 'unknown'}")
            else:
                logger.warning("QueryAnalyzer initialized without embedding model")
        
        # The reference query that represents the current topic
        self.reference_topic = None
        
        # Default threshold - lower for more sensitivity to connection between queries
        self.default_threshold = threshold
        
        # Stopwords to filter out for better matching
        self.stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
            "with", "by", "is", "are", "was", "were", "am", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "can", "could", "will", 
            "would", "shall", "should", "may", "might", "must", "of", "from", "about",
            "as", "if", "when", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "my", "your", "his", "her",
            "its", "our", "their", "this", "that", "these", "those", "i", "you", "he",
            "she", "it", "we", "they", "me", "him", "them", "please", "help"
        }
    
    def _tokenize(self, text):
        """Tokenize text into meaningful words (removing stopwords)"""
        tokens = text.lower().split()
        return [token for token in tokens if token not in self.stopwords]
    
    def _get_ngrams(self, tokens, n=2):
        """Generate n-grams from a list of tokens"""
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens) >= n else []
    
    def _term_overlap_score(self, tokens1, tokens2):
        """Calculate term overlap based on Jaccard similarity of tokens"""
        if not tokens1 or not tokens2:
            return 0.0
            
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        return len(set1.intersection(set2)) / len(set1.union(set2))
    
    def _term_frequency_similarity(self, tokens1, tokens2):
        """Calculate similarity based on term frequency distributions"""
        if not tokens1 or not tokens2:
            return 0.0
            
        # Count frequencies
        counter1 = Counter(tokens1)
        counter2 = Counter(tokens2)
        
        # Get all unique terms
        all_terms = set(counter1.keys()).union(set(counter2.keys()))
        
        # Create frequency vectors
        vec1 = np.array([counter1.get(term, 0) for term in all_terms])
        vec2 = np.array([counter2.get(term, 0) for term in all_terms])
        
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            vec1_norm = vec1 / norm1
            vec2_norm = vec2 / norm2
            
            # Cosine similarity between frequency vectors
            return float(np.dot(vec1_norm, vec2_norm))
        
        return 0.0

    def _get_embedding_openai(self, text: str) -> np.ndarray:
        """Get embedding using OpenAI API"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.openai_embedding_model,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {str(e)}")
            # Return zero vector as fallback
            embedding_dim = 1536 if "small" in self.openai_embedding_model else 3072
            return np.zeros(embedding_dim)

    def _get_embedding_sentence_transformer(self, text: str) -> np.ndarray:
        """Get embedding using SentenceTransformer"""
        try:
            return self.embedding_model.encode([text])[0]
        except Exception as e:
            logger.error(f"SentenceTransformer embedding failed: {str(e)}")
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using the configured method"""
        if self.use_openai_embeddings:
            return self._get_embedding_openai(text)
        else:
            return self._get_embedding_sentence_transformer(text)
    
    def calculate_similarity(self, query1: str, query2: str) -> Tuple[float, Dict[str, float]]:
        """Calculate similarity between two queries using multiple metrics"""
        # Normalize queries
        query1 = query1.lower().strip()
        query2 = query2.lower().strip()
        
        # Quick check for identical queries
        if query1 == query2:
            return 1.0, {"embedding": 1.0, "term_overlap": 1.0, "ngram_overlap": 1.0}
        
        # Initialize similarity scores dictionary
        similarity_scores = {}
        
        # Tokenize (removing stopwords)
        tokens1 = self._tokenize(query1)
        tokens2 = self._tokenize(query2)
        
        # 1. Embedding similarity (semantic similarity)
        if self.model_available:
            try:
                embedding1 = self._get_embedding(query1)
                embedding2 = self._get_embedding(query2)
                
                # Calculate cosine similarity
                similarity_scores['embedding'] = float(
                    cosine_similarity([embedding1], [embedding2])[0][0]
                )
            except Exception as e:
                logger.error(f"Embedding similarity calculation failed: {str(e)}")
                similarity_scores['embedding'] = 0.0
        else:
            similarity_scores['embedding'] = 0.0
        
        # 2. Term overlap (Jaccard similarity)
        similarity_scores['term_overlap'] = self._term_overlap_score(tokens1, tokens2)
        
        # 3. N-gram overlap for phrases
        bigrams1 = set(self._get_ngrams(tokens1, 2))
        bigrams2 = set(self._get_ngrams(tokens2, 2))
        
        if bigrams1 and bigrams2:
            similarity_scores['ngram_overlap'] = len(bigrams1.intersection(bigrams2)) / len(bigrams1.union(bigrams2))
        else:
            similarity_scores['ngram_overlap'] = 0.0
        
        # 4. Term frequency similarity
        similarity_scores['term_frequency'] = self._term_frequency_similarity(tokens1, tokens2)
        
        # 5. Length similarity
        len_ratio = min(len(tokens1), len(tokens2)) / max(len(tokens1), len(tokens2)) if max(len(tokens1), len(tokens2)) > 0 else 0
        similarity_scores['length_ratio'] = len_ratio
        
        # Combine scores based on whether embedding model is available
        if self.model_available:
            combined_score = (
                0.50 * similarity_scores['embedding'] +       # Semantic similarity
                0.25 * similarity_scores['term_overlap'] +    # Word overlap
                0.15 * similarity_scores['ngram_overlap'] +   # Phrase overlap
                0.10 * similarity_scores['term_frequency']    # Distribution similarity
            )
        else:
            # Without embeddings, rely more on term-based metrics
            combined_score = (
                0.50 * similarity_scores['term_overlap'] +    # Word overlap
                0.30 * similarity_scores['ngram_overlap'] +   # Phrase overlap
                0.20 * similarity_scores['term_frequency']    # Distribution similarity
            )
        
        return combined_score, similarity_scores
    
    def analyze_query(self, query, threshold=None, verbose=False):
        """Determine if a query is a new topic or follow-up"""
        start_time = time.time()
        
        # Use default threshold if none provided
        if threshold is None:
            threshold = self.default_threshold
        
        # Initialize result
        result = {
            "query": query,
            "type": "new",
            "confidence": 1.0,
            "similarity_score": 0
        }
        
        # First query is always new
        if self.reference_topic is None:
            result["explanation"] = "First query in conversation"
            self.reference_topic = query
            
            if verbose:
                print(f"QUERY: '{query}'")
                print("RESULT: New query (first in conversation)")
        else:
            # Calculate similarity to reference topic
            similarity_score, detailed_scores = self.calculate_similarity(
                self.reference_topic, query
            )
            
            result["similarity_score"] = similarity_score
            result["detailed_scores"] = detailed_scores
            result["threshold"] = threshold
            
            # Store current reference for reporting
            current_reference = self.reference_topic
            
            # Determine if query is a follow-up
            if similarity_score >= threshold:
                result["type"] = "follow_up"
                result["confidence"] = min(1.0, (similarity_score - threshold) * 3 + 0.6)
                result["explanation"] = f"Query is related to current topic: '{current_reference}'"
                
                # DON'T update reference_topic for follow-ups
                
                if verbose:
                    print(f"QUERY: '{query}'")
                    print(f"CURRENT TOPIC: '{current_reference}'")
                    print(f"RESULT: Follow-up query (similarity: {similarity_score:.4f})")
                    print(f"Confidence: {result['confidence']:.2f}")
                    print(f"Threshold: {threshold:.2f}")
                    print(f"Detailed scores: {detailed_scores}")
            else:
                # Update reference_topic for new topics
                old_reference = self.reference_topic
                self.reference_topic = query
                
                result["confidence"] = min(1.0, (threshold - similarity_score) * 3 + 0.6)
                result["explanation"] = f"New topic (previous topic was: '{old_reference}')"
                
                if verbose:
                    print(f"QUERY: '{query}'")
                    print(f"PREVIOUS TOPIC: '{old_reference}'")
                    print(f"RESULT: New topic (similarity: {similarity_score:.4f})")
                    print(f"Confidence: {result['confidence']:.2f}")
                    print(f"Threshold: {threshold:.2f}")
                    print(f"Detailed scores: {detailed_scores}")
        
        # Calculate processing time
        result["processing_time_ms"] = (time.time() - start_time) * 1000
        
        if verbose:
            print(f"Processing time: {result['processing_time_ms']:.2f}ms")
        
        return result


# Example usage
# if __name__ == "__main__":
#     # Lower threshold for higher sensitivity to related queries
#     analyzer = QueryAnalyzer(
#         embedding_model_path="./models/all-MiniLM-L6-v2",  # Path to your locally saved model
#         threshold=0.25,
#         use_local=True  # This flag tells the class to load from local path
#     )
    
#     while True:
#         query = input("Enter the query: ")
#         result = analyzer.analyze_query(query, verbose=False)
#         print(result)
    