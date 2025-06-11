import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import sqlite3
from typing import List, Dict, Tuple
import os
import textwrap # For better prompt formatting

# === Configuration (Must match Part 1) ===
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "sqlite_texts"
EMBEDDINGS_BASE_DIR = r"E:\Generate SQL Queries\Embeddings"
SQLITE_FILE = r"E:\Generate SQL Queries\SQL Databases\sakila.db"
# IMPORTANT: Replace with the actual path to your downloaded LLM
LLM_MODEL_PATH = r"E:\Generate SQL Queries\prem (LLM)"

# Derive db_name and embedding_dir_path as in Part 1
db_name = os.path.splitext(os.path.basename(SQLITE_FILE))[0]
embedding_dir_path = os.path.join(EMBEDDINGS_BASE_DIR, db_name)

# === Step 1: Load local ChromaDB Collection ===
print(f"Initializing embedding function with model: {EMBEDDING_MODEL_NAME}...")
embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

print(f"Initializing ChromaDB client at: {embedding_dir_path}...")
client = chromadb.PersistentClient(path=embedding_dir_path)

print(f"Getting ChromaDB collection: {CHROMA_COLLECTION_NAME}...")
try:
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME, embedding_function=embedding_function)
    print(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' loaded successfully.")
    print(f"Total documents in collection: {collection.count()}")
except Exception as e:
    print(f"Error loading ChromaDB collection: {e}")
    print("Please ensure you have run Part 1 (embedding generation) first.")
    exit()

# === Step 2: Load your Local SQL-Tuned LLM ===
print(f"\nLoading LLM from: {LLM_MODEL_PATH}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, local_files_only=True)
    # Use bfloat16 for GPU memory efficiency if available, otherwise default to float32
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, local_files_only=True,
                                                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None)
    tokenizer.pad_token = tokenizer.eos_token # Ensure pad_token exists

    sql_llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1, # Use GPU if available (0 is first GPU)
        do_sample=False,        # Set to False for deterministic (less creative) output
        # Removed temperature and top_p here as they are ignored when do_sample=False
        eos_token_id=tokenizer.eos_token_id, # Ensure generation stops at EOS token
        max_new_tokens=256,     # Limit generated tokens to prevent rambling
        pad_token_id=tokenizer.pad_token_id
    )
    print("LLM loaded successfully.")
except Exception as e:
    print(f"Error loading LLM: {e}")
    print("Please ensure the LLM_MODEL_PATH is correct and the model files are complete.")
    exit()

# === Step 3: Database Schema Extraction for Detailed Context ===
def get_db_schema_for_llm(db_path: str) -> Dict[str, Dict]:
    """
    Extracts detailed schema information (tables, columns, primary/foreign keys)
    from an SQLite database, formatted as CREATE TABLE statements for the LLM.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    schema_info = {}

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]

    for table_name in tables:
        table_schema_statement = f"CREATE TABLE {table_name} (\n"
        columns = []
        cursor.execute(f"PRAGMA table_info({table_name});")
        for col_info in cursor.fetchall():
            col_name = col_info[1]
            col_type = col_info[2]
            pk = " PRIMARY KEY" if col_info[5] else ""
            not_null = " NOT NULL" if col_info[3] else ""
            dflt_value = f" DEFAULT {col_info[4]}" if col_info[4] is not None else ""
            columns.append(f"  {col_name} {col_type}{not_null}{pk}{dflt_value}")
        table_schema_statement += ",\n".join(columns)

        # Add foreign keys
        fks = []
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        for fk_info in cursor.fetchall():
            # (id, seq, from_table, from_col, to_col, on_update, on_delete, match)
            # In SQLite PRAGMA, 'from_table' column (fk_info[2]) is actually the target table name
            # and 'to_col' (fk_info[4]) is the target column name.
            from_col = fk_info[3]
            target_table = fk_info[2]
            target_col = fk_info[4]
            fks.append(f"  FOREIGN KEY ({from_col}) REFERENCES {target_table}({target_col})")
        
        if fks:
            table_schema_statement += ",\n" + ",\n".join(fks)

        table_schema_statement += "\n);"
        schema_info[table_name] = {"create_statement": table_schema_statement}

    conn.close()
    return schema_info

print("\nExtracting database schema for detailed context...")
full_db_schema = get_db_schema_for_llm(SQLITE_FILE)
print("Database schema extracted.")

# === Step 4: Query ChromaDB for Relevant Context ===
def retrieve_context(question: str, top_k=10) -> List[str]:
    """
    Retrieves relevant schema and data snippets from ChromaDB,
    prioritizing full schema statements for relevant tables.
    """
    results = collection.query(query_texts=[question], n_results=top_k, include=['documents', 'metadatas'])
    
    formatted_contexts = []
    
    # Collect unique table names from retrieved items
    relevant_tables = set()
    if results["metadatas"] and results["metadatas"][0]:
        for meta in results["metadatas"][0]:
            if 'table' in meta:
                relevant_tables.add(meta['table'])

    # Add full schema for relevant tables first
    schema_context_added = set()
    for table_name in relevant_tables:
        if table_name in full_db_schema and table_name not in schema_context_added:
            formatted_contexts.append(full_db_schema[table_name]["create_statement"])
            schema_context_added.add(table_name)
    
    # Add retrieved data snippets (if any)
    if results["documents"] and results["documents"][0]:
        for i, doc_text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            if meta.get("type") == "value": # Only add data values
                # Ensure the original text from ChromaDB is used, not the embedding text directly
                # The format "Data in table.column: value" was used in Part 1
                value_part = doc_text.split(':', 1)[-1].strip()
                formatted_contexts.append(f"-- Example data for {meta['table']}.{meta['column']}: {value_part}")
            elif meta.get("type") == "schema" and meta['table'] not in schema_context_added:
                # Fallback: if a schema snippet was retrieved but not yet added as full CREATE TABLE statement
                formatted_contexts.append(full_db_schema[meta['table']]["create_statement"])
                schema_context_added.add(meta['table'])

    return formatted_contexts

# === Step 5: Generate SQL Using Local LLM ===
# A few-shot example can significantly improve performance.
# These examples should be carefully chosen and represent common query patterns.
# Customize these examples based on your database's structure and typical queries.
FEW_SHOT_EXAMPLES = textwrap.dedent("""
### Example 1:
### Database Schema:
CREATE TABLE actor (
  actor_id INTEGER PRIMARY KEY,
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);
CREATE TABLE film (
  film_id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT,
  release_year INTEGER,
  rental_duration INTEGER NOT NULL,
  rental_rate NUMERIC(4,2) NOT NULL,
  length INTEGER,
  replacement_cost NUMERIC(5,2) NOT NULL,
  rating TEXT DEFAULT 'G',
  last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
  special_features TEXT,
  fulltext TEXT
);
CREATE TABLE film_actor (
  actor_id INTEGER NOT NULL,
  film_id INTEGER NOT NULL,
  last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
  PRIMARY KEY (actor_id, film_id),
  FOREIGN KEY (actor_id) REFERENCES actor(actor_id),
  FOREIGN KEY (film_id) REFERENCES film(film_id)
);
### Context (Data Examples):
-- Example data for actor.first_name: PENELOPE
-- Example data for actor.last_name: GUINESS
-- Example data for film.title: ACADEMY DINOSAUR
### User Request:
How many films did the actor 'PENELOPE GUINESS' appear in?
### SQL Query:
SELECT
  COUNT(T1.film_id)
FROM film_actor AS T1
INNER JOIN actor AS T2
  ON T1.actor_id = T2.actor_id
WHERE
  T2.first_name = 'PENELOPE' AND T2.last_name = 'GUINESS';

### Example 2:
### Database Schema:
CREATE TABLE category (
  category_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);
CREATE TABLE film_category (
  film_id INTEGER NOT NULL,
  category_id INTEGER NOT NULL,
  last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
  PRIMARY KEY (film_id, category_id),
  FOREIGN KEY (film_id) REFERENCES film(film_id),
  FOREIGN KEY (category_id) REFERENCES category(category_id)
);
CREATE TABLE film (
  film_id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT,
  release_year INTEGER,
  rental_duration INTEGER NOT NULL,
  rental_rate NUMERIC(4,2) NOT NULL,
  length INTEGER,
  replacement_cost NUMERIC(5,2) NOT NULL,
  rating TEXT DEFAULT 'G',
  last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
  special_features TEXT,
  fulltext TEXT
);
### Context (Data Examples):
-- Example data for category.name: Action
-- Example data for category.name: Comedy
### User Request:
Show me all films in the 'Action' category.
### SQL Query:
SELECT
  T1.title
FROM film AS T1
INNER JOIN film_category AS T2
  ON T1.film_id = T2.film_id
INNER JOIN category AS T3
  ON T2.category_id = T3.category_id
WHERE
  T3.name = 'Action';

### Example 3:
### Database Schema:
CREATE TABLE customer (
  customer_id INTEGER PRIMARY KEY,
  store_id INTEGER NOT NULL,
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  email TEXT,
  address_id INTEGER NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  create_date TIMESTAMP NOT NULL,
  last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
  FOREIGN KEY (address_id) REFERENCES address(address_id)
);
### Context (Data Examples):
-- Example data for customer.first_name: MARY
-- Example data for customer.last_name: SMITH
-- Example data for customer.email: MARY.SMITH@sakilacustomer.org
### User Request:
What is the email address of the customer named Mary Smith?
### SQL Query:
SELECT
  email
FROM customer
WHERE
  first_name = 'MARY' AND last_name = 'SMITH';
""")

def generate_sql_from_question(question: str) -> str:
    context_snippets = retrieve_context(question)
    
    # Construct the context block, ensuring schema is clearly separated from data examples
    schema_parts = []
    data_example_parts = []
    for snippet in context_snippets:
        if snippet.startswith("CREATE TABLE"):
            schema_parts.append(snippet)
        elif snippet.startswith("-- Example data"):
            data_example_parts.append(snippet)

    schema_block = "\n".join(schema_parts) if schema_parts else "No relevant schema found."
    data_example_block = "\n".join(data_example_parts) if data_example_parts else "No relevant data examples found."

    prompt = f"""You are an expert SQLite SQL assistant. Your goal is to generate precise, correct, and executable SQL queries given a user's natural language request and the database schema.
Only generate SELECT queries. Do not generate any explanations or additional text, just the SQL query.
If you cannot generate a meaningful query, respond with 'NO_QUERY_POSSIBLE'.

### Database Schema:
{schema_block}

### Context (Data Examples):
{data_example_block}

{FEW_SHOT_EXAMPLES}

### User Request:
{question}

### SQL Query:"""

    # For debugging the prompt sent to LLM (uncomment to see the full prompt)
    # print(f"\n--- Prompt sent to LLM (for debugging) ---\n{prompt}\n--- End Prompt ---")

    outputs = sql_llm(prompt) # Generation parameters are now set in the pipeline init
    
    generated_text = outputs[0]["generated_text"]
    
    # Extract only the SQL Query part, robustly
    sql_part = generated_text.split("### SQL Query:")[-1].strip()
    
    # Clean up any trailing text that might be generated by the LLM
    if "###" in sql_part:
        sql_part = sql_part.split("###")[0].strip()
    
    # Ensure it ends with a semicolon and remove any other punctuation if present
    if sql_part.endswith('.'):
        sql_part = sql_part[:-1].strip()
    if not sql_part.endswith(';'):
        sql_part += ';'
    
    if "NO_QUERY_POSSIBLE" in sql_part.upper():
        return "NO_QUERY_POSSIBLE"

    return sql_part

# === Step 6: SQL Execution and Validation ===
def execute_sql_query(sql_query: str, db_path: str = SQLITE_FILE) -> Tuple[List[Tuple], List[str]]:
    """
    Executes a SQL query against the SQLite database and returns results and column names.
    Raises ValueError if the query is not a SELECT statement.
    """
    conn = None
    results = []
    column_names = []
    try:
        # Basic validation: ensure it's a SELECT query
        if not sql_query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed for execution by this system.")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        
        if cursor.description: # Check if there are columns (for SELECT queries)
            column_names = [description[0] for description in cursor.description]
            results = cursor.fetchall()
        
        return results, column_names
    except sqlite3.Error as e:
        print(f"Database error executing SQL: {e}")
        raise # Re-raise to be caught by the main loop for error display
    except ValueError as e:
        print(f"SQL Validation Error: {e}")
        raise
    finally:
        if conn:
            conn.close()

# === Step 7: Command Line Interface for Air-Gapped Usage ===
if __name__ == "__main__":
    print("\n🛡️ Air-Gapped SQL Chatbot Initialized (Offline Mode)")
    print("--------------------------------------------------")
    print("Type your natural language question to get a SQL query and its results.")
    print("Type 'exit' to quit.")
    print("--------------------------------------------------")

    while True:
        user_question = input("\n🔍 Enter your question: ")
        if user_question.lower() == 'exit':
            break
        
        print("\nWorking on your request...")
        try:
            generated_sql = generate_sql_from_question(user_question)
            
            if generated_sql == "NO_QUERY_POSSIBLE":
                print("\n❌ Could not generate a meaningful SQL query for your request. Please try rephrasing.")
                continue

            print("\n✅ Generated SQL Query:\n", generated_sql)
            
            print("\nExecuting SQL query...")
            query_results, col_names = execute_sql_query(generated_sql)
            
            if col_names:
                print("\n📊 Query Results:")
                # Print header
                print(" | ".join(col_names))
                print("-" * (sum(len(c) for c in col_names) + (len(col_names) - 1) * 3))
                # Print rows
                for row in query_results:
                    print(" | ".join(map(str, row)))
                print(f"\nTotal rows returned: {len(query_results)}")
            else:
                print("\nℹ️ Query executed successfully, but returned no results.")

        except Exception as e:
            print(f"❌ An error occurred during SQL generation or execution: {e}")
            print("Please try rephrasing your question or check the console for details.")
