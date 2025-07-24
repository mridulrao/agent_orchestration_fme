import os
import pandas as pd
from lancedb import connect
import pickle

def extract_all_documents(db_uri, table_name, output_file):
    """
    Extract all documents from a LanceDB table and save them to a pickle file.
    
    Args:
        db_uri: URI to the LanceDB database
        table_name: Name of the table to extract documents from
        output_file: File path to save the extracted documents
        
    Returns:
        Dictionary containing all documents
    """
    try:
        # Connect to LanceDB
        print(f"Connecting to LanceDB at: {db_uri}")
        db = connect(db_uri)
        
        # Check if table exists
        if table_name not in db.table_names():
            print(f"Table '{table_name}' does not exist in the database.")
            print(f"Available tables: {', '.join(db.table_names())}")
            return {}
        
        # Open the table
        table = db.open_table(table_name)
        
        # Get all data from the table
        all_data = table.to_pandas()
        
        # Create documents dictionary
        documents = {}
        
        # Extract all documents
        for _, row in all_data.iterrows():
            # Extract document IDs from the correct columns
            sys_id = row.get('sys_id', '')
            kb_document_id = row.get('kb_document_id', '')
            
            # Use kb_document_id as the main identifier, fallback to sys_id if not available
            doc_id = kb_document_id if kb_document_id else sys_id
            
            # Skip if no valid ID found
            if not doc_id:
                continue
                
            # Extract title and content
            title = row.get('title', '')
            content = row.get('contents', '')
            
            # Handle bytes content if needed
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')
            
            # Store in dictionary with doc_id as key
            documents[doc_id] = {
                'document_id': doc_id,
                'title': title,
                'content': content
            }
        
        # Print results summary
        print(f"Extracted {len(documents)} documents from the table.")
        
        # Save to pickle file
        if documents:
            with open(output_file, 'wb') as f:
                pickle.dump(documents, f)
            print(f"Saved {len(documents)} documents to {output_file}")
        else:
            print("No documents found to save.")
            
        return documents
        
    except Exception as e:
        print(f"Error extracting documents from LanceDB table: {str(e)}")
        return {}

if __name__ == "__main__":
    # Configuration - modify these values as needed
    DB_URI = os.environ.get('KB_LANCEDB_URI', './simple_vector_index')
    TABLE_NAME = 'servicenow_knowledge_simple'
    OUTPUT_FILE = 'all_documents.pickle'
    
    # Extract all documents
    extract_all_documents(DB_URI, TABLE_NAME, OUTPUT_FILE)