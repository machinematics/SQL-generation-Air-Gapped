import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import sqlite3
from datetime import datetime
from tqdm import tqdm
import os
from typing import Dict, List, Tuple

# === Configuration ===
# Define the path to your SQLite database
SQLITE_FILE = r"E:\Generate SQL Queries\SQL Databases\sakila.db"
# Define the base directory where embeddings will be stored
EMBEDDINGS_BASE_DIR = r"E:\Generate SQL Queries\Embeddings"
# Name of the SentenceTransformer model to use for embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # This model will be downloaded locally if not present
# Name of the ChromaDB collection
CHROMA_COLLECTION_NAME = "sqlite_texts"
# Batch size for adding documents to ChromaDB
CHROMA_BATCH_SIZE = 64

# === Helper Function: Extract Detailed Schema ===
def get_db_schema_for_embedding(db_path: str) -> List[Dict]:
    """
    Extracts detailed schema information (tables, columns, primary/foreign keys)
    from an SQLite database, formatted for embedding.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    schema_docs = []

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]

    for table_name in tables:
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns_info = cursor.fetchall()
        
        column_definitions = []
        for col_info in columns_info:
            col_name = col_info[1]
            col_type = col_info[2]
            pk = " PRIMARY KEY" if col_info[5] else ""
            not_null = " NOT NULL" if col_info[3] else ""
            # default_value = f" DEFAULT {col_info[4]}" if col_info[4] is not None else "" # Omitted for brevity in embedding text
            column_definitions.append(f"{col_name} {col_type}{not_null}{pk}")

        # Get foreign key info
        fks_info = []
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        for fk_info in cursor.fetchall():
            # (id, seq, from_table, from_col, to_col, on_update, on_delete, match)
            from_col = fk_info[3]
            to_table = fk_info[2] # In SQLite FK list, 'from_table' column actually means the target table!
            to_col = fk_info[4]
            fks_info.append(f"FOREIGN KEY ({from_col}) REFERENCES {to_table}({to_col})")
        
        # Combine into a schema text for embedding
        schema_text = f"Table: {table_name}\nColumns: " + ", ".join(column_definitions)
        if fks_info:
            schema_text += "\nForeign Keys: " + "; ".join(fks_info)
        
        schema_docs.append({
            "document": schema_text,
            "id": f"{table_name}_schema",
            "metadata": {"type": "schema", "table": table_name}
        })
    
    conn.close()
    return schema_docs

# === Step 1: Initialize Embedding Function ===
print(f"Initializing embedding function with model: {EMBEDDING_MODEL_NAME}...")
# This will download the model to your local Hugging Face cache (~90MB for all-MiniLM-L6-v2)
embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

# === Step 2: Define SQLite DB path and derive DB name ===
db_name = os.path.splitext(os.path.basename(SQLITE_FILE))[0] # e.g., "sakila"
embedding_dir_path = os.path.join(EMBEDDINGS_BASE_DIR, db_name)

# Ensure the embedding directory exists
os.makedirs(embedding_dir_path, exist_ok=True)
print(f"Embeddings will be stored in: {embedding_dir_path}")

# === Step 3: Initialize Persistent Chroma Client ===
print(f"Initializing ChromaDB client at: {embedding_dir_path}...")
client = chromadb.PersistentClient(path=embedding_dir_path)

# === Step 4: Connect to SQLite Database ===
print(f"Connecting to SQLite database: {SQLITE_FILE}...")
conn = sqlite3.connect(SQLITE_FILE)
cursor = conn.cursor()

# === Step 5: Get All Table Names (for data extraction) ===
print("Fetching table names from the database...")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [table[0] for table in cursor.fetchall()]
print(f"Found {len(tables)} tables: {', '.join(tables)}")

# === Step 6: Prepare Documents, IDs, and Metadata lists ===
documents_to_add = []
ids_to_add = []
metadatas_to_add = []

# === Step 7: Embed Detailed Schema Definitions ===
print("\n--- Embedding Detailed Schema Definitions ---")
detailed_schema_docs = get_db_schema_for_embedding(SQLITE_FILE)
for doc_info in tqdm(detailed_schema_docs, desc="Processing detailed schema for embedding"):
    documents_to_add.append(doc_info["document"])
    ids_to_add.append(doc_info["id"])
    metadatas_to_add.append(doc_info["metadata"])

# === Step 8: Embed TEXT/CHAR Data Entries ===
print("\n--- Embedding TEXT/CHAR Data Entries ---")
for table_name in tables:
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()

    for column in columns:
        column_name = column[1]
        col_type = column[2].upper()

        # Check for common text-based column types
        if "TEXT" in col_type or "CHAR" in col_type or "VARCHAR" in col_type:
            print(f"  Processing data for table: {table_name}, column: {column_name}")

            try:
                # Use a LIMIT for very large tables if full data embedding is too much
                cursor.execute(f"SELECT rowid, {column_name} FROM {table_name} LIMIT 5000;") # Limit to prevent excessive data
                rows = cursor.fetchall()
            except sqlite3.OperationalError as e:
                print(f"    Skipping column {column_name} in table {table_name} due to error: {e}")
                continue # Skip to the next column if there's an issue

            for row in tqdm(rows, desc=f"  {table_name}.{column_name}", unit="rows", leave=False):
                row_id_val = row[0]
                raw_text = row[1]

                if raw_text is not None and str(raw_text).strip() != "":
                    # Add context to the document
                    contextual_text = f"Data in {table_name}.{column_name}: {str(raw_text).strip()}"
                    documents_to_add.append(contextual_text)
                    ids_to_add.append(f"{table_name}_{column_name}_{row_id_val}")
                    metadatas_to_add.append({
                        "type": "value",
                        "table": table_name,
                        "column": column_name,
                        "rowid": row_id_val
                    })
conn.close()
print("\nFinished extracting data from SQLite database.")


# === Step 9: Create or Load Chroma Collection ===
print(f"\nCreating or loading ChromaDB collection: {CHROMA_COLLECTION_NAME}...")
collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    embedding_function=embedding_function, # Crucial: use the same function
    metadata={
        "description": f"Embedded texts and schemas from {db_name} SQLite DB",
        "created": str(datetime.now())
    }
)

# === Step 10: Batched Insertion into Chroma ===
def batch_add_to_chroma(collection_obj, docs, ids, metas, batch_size):
    if not docs:
        print("No documents to add to ChromaDB.")
        return

    print("Checking for existing documents in ChromaDB to avoid re-adding (this might take a while for large collections)...")
    # Optimize checking for existing IDs by querying in batches too if 'ids' list is very large
    existing_ids_in_db = set()
    for i in tqdm(range(0, len(ids), batch_size), desc="Checking existing IDs"):
        batch_ids = ids[i:i + batch_size]
        try:
            existing_ids_in_db.update(collection_obj.get(ids=batch_ids, include=[])['ids'])
        except Exception as e:
            print(f"Warning: Could not retrieve existing IDs for batch starting with {batch_ids[0]}. Error: {e}")
            # In case of error, assume none exist for this batch to attempt adding
            pass 

    new_docs_to_add = []
    new_ids_to_add = []
    new_metadatas_to_add = []

    for i, doc_id in enumerate(ids):
        if doc_id not in existing_ids_in_db:
            new_docs_to_add.append(docs[i])
            new_ids_to_add.append(ids[i])
            new_metadatas_to_add.append(metas[i])

    if not new_docs_to_add:
        print("All documents already exist in ChromaDB or no new documents to add. Skipping insertion.")
        return

    print(f"Adding {len(new_docs_to_add)} new documents to ChromaDB...")
    for i in tqdm(range(0, len(new_docs_to_add), batch_size), desc="Storing new embeddings", unit="batch"):
        try:
            collection_obj.add(
                documents=new_docs_to_add[i:i + batch_size],
                ids=new_ids_to_add[i:i + batch_size],
                metadatas=new_metadatas_to_add[i:i + batch_size]
            )
        except Exception as e:
            print(f"Error adding batch starting with ID {new_ids_to_add[i]}: {e}")
            # Depending on error, you might want to skip or retry

batch_add_to_chroma(collection, documents_to_add, ids_to_add, metadatas_to_add, CHROMA_BATCH_SIZE)

# === Step 11: Sanity Check ===
print("\n--- ChromaDB Sanity Check ---")
print("✅ First 10 Records:")
print(collection.peek())
print("✅ Total Records Stored:", collection.count())
print("\nPart 1: Local embeddings generation complete. ChromaDB is populated.")
