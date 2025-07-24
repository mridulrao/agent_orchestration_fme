import pickle
import argparse
import sys
import re
import os
import shutil
from tabulate import tabulate

def load_documents(pickle_file):
    """
    Load documents from a pickle file.
    
    Args:
        pickle_file: Path to the pickle file containing documents dictionary
        
    Returns:
        Dictionary of documents
    """
    try:
        with open(pickle_file, 'rb') as f:
            documents = pickle.load(f)
        print(f"Loaded {len(documents)} documents from {pickle_file}")
        return documents
    except Exception as e:
        print(f"Error loading documents from {pickle_file}: {str(e)}")
        return None

def save_documents(documents, pickle_file, backup=True):
    """
    Save documents back to pickle file with optional backup.
    
    Args:
        documents: Dictionary of documents to save
        pickle_file: Path to the pickle file
        backup: Whether to create a backup before saving
    """
    try:
        if backup and os.path.exists(pickle_file):
            backup_file = pickle_file + '.backup'
            shutil.copy(pickle_file, backup_file)
            print(f"Created backup: {backup_file}")
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(documents, f)
        print(f"Documents saved to {pickle_file}")
        return True
    except Exception as e:
        print(f"Error saving documents to {pickle_file}: {str(e)}")
        return False

def search_by_id(documents, doc_id):
    """
    Search for a document by exact ID.
    
    Args:
        documents: Dictionary of documents
        doc_id: Document ID to search for
        
    Returns:
        Document dictionary if found, None otherwise
    """
    if doc_id in documents:
        return documents[doc_id]
    return None

def search_by_partial_id(documents, partial_id):
    """
    Search for documents where ID contains the partial_id string.
    
    Args:
        documents: Dictionary of documents
        partial_id: Partial document ID to search for
        
    Returns:
        List of (document_id, document) tuples for matching documents
    """
    matches = []
    for doc_id, doc in documents.items():
        if partial_id.lower() in doc_id.lower():
            matches.append((doc_id, doc))
    return matches

def search_by_content(documents, query, case_sensitive=False):
    """
    Search for documents where content contains the query string.
    
    Args:
        documents: Dictionary of documents
        query: Text to search for in document content
        case_sensitive: Whether to perform case-sensitive search
        
    Returns:
        List of (document_id, document) tuples for matching documents
    """
    matches = []
    
    # Compile regex pattern for search
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(re.escape(query), flags)
    
    for doc_id, doc in documents.items():
        content = doc.get('content', '')
        if pattern.search(content):
            matches.append((doc_id, doc))
    
    return matches

def display_document(doc_id, document, show_full_content=False):
    """
    Display document information.
    
    Args:
        doc_id: Document ID
        document: Document dictionary
        show_full_content: Whether to show the full content
    """
    print(f"\nDocument ID: {doc_id}")
    print(f"KB ID: {document.get('document_id', 'N/A')}")
    print(f"Title: {document.get('title', 'N/A')}")
    
    content = document.get('content', '')
    if content:
        if show_full_content:
            print("\nContent:")
            print("=" * 80)
            print(content)
            print("=" * 80)
        else:
            # Show a preview (first 200 characters)
            content_preview = content[:200] + '...' if len(content) > 200 else content
            print(f"\nContent Preview: {content_preview}")
            print("\nUse --full to see the complete content.")

def main():
    parser = argparse.ArgumentParser(description='Search and modify documents from pickle file')
    parser.add_argument('--file', type=str, required=True,
                        help='Path to the pickle file containing documents')
    parser.add_argument('--id', type=str, 
                        help='Search for document by exact ID')
    parser.add_argument('--partial-id', type=str,
                        help='Search for documents by partial ID match')
    parser.add_argument('--content', type=str,
                        help='Search for documents by content text')
    parser.add_argument('--case-sensitive', action='store_true',
                        help='Make content search case-sensitive')
    parser.add_argument('--full', action='store_true',
                        help='Show full document content')
    parser.add_argument('--list', action='store_true',
                        help='List all document IDs')
    
    args = parser.parse_args()
    
    # Load documents
    documents = load_documents(args.file)
    
    # List all document IDs
    if args.list:
        print("\nAll Document IDs:")
        for i, doc_id in enumerate(documents.keys(), 1):
            print(f"{i}. {doc_id}")
        sys.exit(0)
    
    # Search by exact ID
    if args.id:
        document = search_by_id(documents, args.id)
        if document:
            display_document(args.id, document, args.full)
        else:
            print(f"No document found with ID: {args.id}")
        sys.exit(0)
    
    # Search by partial ID
    if args.partial_id:
        matches = search_by_partial_id(documents, args.partial_id)
        if matches:
            print(f"\nFound {len(matches)} documents matching partial ID '{args.partial_id}':")
            for i, (doc_id, doc) in enumerate(matches, 1):
                print(f"\n{i}. {doc_id}")
                if args.full:
                    display_document(doc_id, doc, True)
        else:
            print(f"No documents found with partial ID: {args.partial_id}")
        sys.exit(0)
    
    # Search by content
    if args.content:
        matches = search_by_content(documents, args.content, args.case_sensitive)
        if matches:
            print(f"\nFound {len(matches)} documents containing '{args.content}':")
            
            # Create a table for display
            table_data = []
            for i, (doc_id, doc) in enumerate(matches, 1):
                content = doc.get('content', '')
                # Get a snippet around the matched text
                content_lower = content.lower()
                query_lower = args.content.lower()
                
                # Find the position of the query in the content
                pos = content_lower.find(query_lower)
                if pos >= 0:
                    # Create a snippet around the match
                    start = max(0, pos - 50)
                    end = min(len(content), pos + len(args.content) + 50)
                    snippet = content[start:end]
                    # Add ellipsis if we're not at the start/end
                    if start > 0:
                        snippet = '...' + snippet
                    if end < len(content):
                        snippet = snippet + '...'
                else:
                    snippet = content[:100] + '...' if len(content) > 100 else content
                
                table_data.append([i, doc_id, doc.get('title', 'N/A'), snippet])
            
            # Print the table
            headers = ['#', 'Document ID', 'Title', 'Content Snippet']
            print(tabulate(table_data, headers=headers, tablefmt='grid'))
            
            # Ask if user wants to see full content of a specific document
            if not args.full:
                try:
                    choice = input("\nEnter document number to see full content (or press Enter to skip): ")
                    if choice.strip() and choice.isdigit() and 1 <= int(choice) <= len(matches):
                        doc_id, doc = matches[int(choice) - 1]
                        display_document(doc_id, doc, True)
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user")
        else:
            print(f"No documents found containing: {args.content}")
        sys.exit(0)
    
    # If no search criteria provided, show usage
    parser.print_help()

if __name__ == "__main__":
    main()