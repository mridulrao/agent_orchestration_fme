import os
from pathlib import Path
from lancedb import connect

def debug_lancedb_tables():
    """
    Debug script to see what tables are actually available in your LanceDB
    """
    # Get the correct path
    current_dir = Path(__file__).parent  # rag directory
    project_root = current_dir.parent    # project root  
    lancedb_uri = str(project_root / "indexing" / "fme_vector_index")
    
    print(f"LanceDB URI: {lancedb_uri}")
    print(f"Path exists: {Path(lancedb_uri).exists()}")
    
    # List directory contents
    lancedb_path = Path(lancedb_uri)
    if lancedb_path.exists():
        print("\nDirectory contents:")
        for item in lancedb_path.iterdir():
            print(f"  {item.name} ({'directory' if item.is_dir() else 'file'})")
    
    try:
        # Connect to LanceDB
        db = connect(lancedb_uri)
        
        # Get available table names
        table_names = db.table_names()
        print(f"\nAvailable table names: {table_names}")
        
        if not table_names:
            print("No tables found in LanceDB!")
            print("This suggests the vector index may not have been created properly.")
        else:
            # Try to open each table and get basic info
            for table_name in table_names:
                try:
                    table = db.open_table(table_name)
                    df = table.to_pandas()
                    print(f"\nTable '{table_name}':")
                    print(f"  Rows: {len(df)}")
                    print(f"  Columns: {list(df.columns)}")
                    print(f"  Sample row columns: {df.iloc[0].index.tolist() if len(df) > 0 else 'No data'}")
                except Exception as e:
                    print(f"  Error opening table '{table_name}': {e}")
    
    except Exception as e:
        print(f"Error connecting to LanceDB: {e}")

if __name__ == "__main__":
    debug_lancedb_tables()