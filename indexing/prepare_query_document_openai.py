import pickle
import time
import logging
from typing import List, Dict, Any
from openai import OpenAI
import os

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY", "")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingConverter:
    def __init__(self, openai_api_key: str, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the converter with OpenAI client.
        
        Args:
            openai_api_key: Your OpenAI API key
            embedding_model: OpenAI embedding model to use
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
    
    def embed_query(self, query: str) -> List[float]:
        """
        Create an embedding vector for the query using OpenAI.
        
        Args:
            query: The query to embed
            
        Returns:
            List[float]: The embedding vector
        """
        start_time = time.time()
        
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            
            embedding = response.data[0].embedding
            
            # Log embedding time
            embed_time = time.time() - start_time
            logger.info(f"Created embedding for query in {embed_time:.2f} seconds")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding failed for query: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 1536  # Match dimensionality of text-embedding-3-small
    
    def convert_pickle_file(self, input_file: str, output_file: str) -> None:
        """
        Convert the existing pickle file to use OpenAI embeddings.
        
        Args:
            input_file: Path to the original pickle file
            output_file: Path for the new pickle file with OpenAI embeddings
        """
        try:
            # Load the original pickle file
            logger.info(f"Loading original pickle file: {input_file}")
            with open(input_file, 'rb') as f:
                original_query_documents = pickle.load(f)
            
            logger.info(f"Loaded {len(original_query_documents)} queries")
            
            # Create new dictionary with OpenAI embeddings
            new_query_documents = {}
            
            for i, (query, document_data) in enumerate(original_query_documents.items(), 1):
                logger.info(f"Processing query {i}/{len(original_query_documents)}: {query[:100]}...")
                
                # Create new embedding using OpenAI
                new_embedding = self.embed_query(query)
                
                # Create new document structure with OpenAI embedding
                new_query_documents[query] = {
                    'query_vector': new_embedding,
                    'meta_data': document_data['meta_data'],
                    'query_answer': document_data['query_answer'],
                    'document_title': document_data['document_title'],
                    'document_idc': document_data['document_idc']
                }
                
                # Optional: Add a small delay to avoid rate limiting
                time.sleep(0.1)
            
            # Save the new pickle file
            logger.info(f"Saving new pickle file: {output_file}")
            with open(output_file, 'wb') as f:
                pickle.dump(new_query_documents, f)
            
            logger.info(f"Successfully converted {len(new_query_documents)} queries to OpenAI embeddings")
            
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
            raise
        except Exception as e:
            logger.error(f"Error during conversion: {str(e)}")
            raise

def main():
    """
    Main function to run the conversion.
    """
    # Configuration
    OPENAI_API_KEY = openai_api_key  # Replace with your actual API key
    INPUT_FILE = "query_documents.pickle"
    OUTPUT_FILE = "query_documents_openai.pickle"
    EMBEDDING_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large" for higher quality
    
    # Create converter instance
    converter = EmbeddingConverter(
        openai_api_key=OPENAI_API_KEY,
        embedding_model=EMBEDDING_MODEL
    )
    
    # Convert the pickle file
    try:
        converter.convert_pickle_file(INPUT_FILE, OUTPUT_FILE)
        print(f"Conversion completed successfully!")
        print(f"New file saved as: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Conversion failed: {str(e)}")

if __name__ == "__main__":
    main()