# Natural Language to SQL RAG Pipeline (Air-Gapped)

**Follow us at:** [machinematics](https://github.com/machinematics)

**Project Guide on Medium:** [Natural Language to SQL RAG Pipeline (Air-Gapped)](https://medium.com/@machinematics/building-a-secure-rag-pipeline-for-text-to-sql-generation-in-air-gapped-environments-using-chromadb-2aa47dabfb58)

## Overview

This repository presents a robust, two-part Retrieval Augmented Generation (RAG) pipeline designed to translate natural language questions into executable SQLite SQL queries. A key feature of this solution is its ability to operate in an "air-gapped" (offline) environment once initial models and embeddings are set up, making it ideal for secure and disconnected deployment scenarios.

The pipeline leverages a locally hosted Large Language Model (LLM) augmented by a vector database containing detailed database schema and example data.

## Features

* **Offline Capability:** Fully functional without internet access after initial setup (model download and embedding generation).
* **Semantic Search:** Utilizes vector embeddings for intelligent retrieval of relevant database schema and data snippets.
* **Contextual SQL Generation:** Feeds retrieved context to a local LLM, enabling the generation of accurate and context-aware SQL queries.
* **Schema & Data Embedding:** Transforms SQLite database schemas and relevant textual data into searchable vectors.
* **Robust Query Execution:** Executes generated SQL queries against an SQLite database with built-in validation to ensure only `SELECT` statements are run.
* **Interactive CLI:** A simple command-line interface for easy interaction and testing.

## How It Works

The pipeline operates in two main phases:

### Phase 1: Embedding Generation (Data Preparation)

This phase involves extracting and embedding the database's structural information (schema) and textual data into a vector database.

1.  **Database Schema Extraction:** Detailed `CREATE TABLE` statements (including columns, primary keys, and foreign keys) are extracted from the SQLite database.
2.  **Data Value Extraction:** Relevant textual data from `TEXT`/`CHAR`/`VARCHAR` columns is extracted to provide concrete examples.
3.  **Embedding Generation:** Both the structured schema descriptions and the data values are converted into numerical vector embeddings using a Sentence Transformer model.
4.  **Vector Store Population:** These embeddings, along with their original text and metadata, are stored in a persistent ChromaDB collection.

### Phase 2: Retrieval, Generation, and Execution (Runtime)

This phase handles user queries, retrieves context, generates SQL, and executes it.

1.  **Load Knowledge Base:** The pre-populated ChromaDB collection is loaded, providing access to the embedded schema and data.
2.  **Load Local LLM:** A pre-trained, SQL-tuned Large Language Model is loaded locally using Hugging Face Transformers.
3.  **Context Retrieval:** When a user asks a natural language question, the question is embedded, and a semantic search is performed against ChromaDB to find the most relevant schema definitions and data examples.
4.  **Prompt Construction:** A comprehensive prompt is constructed for the LLM, including:
    * System instructions (e.g., "expert SQLite SQL assistant").
    * The retrieved database schema (`CREATE TABLE` statements).
    * Relevant data examples.
    * Few-shot learning examples (demonstrating desired input/output patterns for SQL generation).
    * The user's natural language question.
5.  **SQL Generation:** The LLM processes the prompt and generates a SQL query based on the provided context and instructions.
6.  **SQL Execution:** The generated SQL query is executed against the SQLite database. A critical validation step ensures that only `SELECT` queries are run for safety.
7.  **Result Presentation:** The results of the SQL query are displayed to the user in a readable format.

## Technologies Used

* **Python 3.x:** The primary programming language.
* **ChromaDB:** The open-source vector database used for storing and querying embeddings.
* **Sentence Transformers:** For generating high-quality text embeddings (`all-MiniLM-L6-v2` model).
* **Hugging Face Transformers:** For loading and running the local Large Language Model (`AutoTokenizer`, `AutoModelForCausalLM`, `pipeline`).
* **PyTorch:** The underlying deep learning framework for the LLM.
* **SQLite3:** The database system used for the knowledge base.
* **tqdm:** For progress bars during long operations.
* **textwrap:** For formatting prompt text.
