# Task-1
# Structured LLM Prompt Pipeline & JSON Extractor

A Python script that turns unstructured text documents (invoices and resumes) into **strictly validated JSON** using an LLM API.

**How it works**

1. **Schemas (Pydantic):** `Invoice` and `Resume` models define exactly what valid output looks like. The invoice model also checks that `subtotal + tax == total`.
2. **Parameterized prompt templates:** each document type has a system instruction, a few-shot example, and a template that inserts the raw document text. Adding a new document type only requires a new entry in the `PROMPTS` dict.
3. **LLM call (OpenRouter):** the script sends the prompt to a chat model through the OpenRouter API, with the JSON Schema included in the system message and `temperature=0` for consistent output.
4. **Validation and retry loop:** the reply is parsed with Pydantic. If validation fails, the error is fed back to the model and it retries (up to 3 times).
5. **Resilience:** HTTP 429/5xx errors and network timeouts are retried with increasing delays.

**Setup**

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Windows PowerShell:
```powershell
$env:OPENROUTER_API_KEY="YOUR_KEY"
$env:OPENROUTER_MODEL="recursive/flint-v1.0:free"
```

Mac/Linux:
```bash
export OPENROUTER_API_KEY="YOUR_KEY"
export OPENROUTER_MODEL=""
```

**Example**

Input (`sample_invoice.txt`):

```
INVOICE No. INV-2041
Date: 12th Oct 2025
Vendor: Mapusa Electronics
Billed to: Rahul Naik

1 x Laptop charger  @ 1500.00 = 1500.00
2 x USB cable       @ 250.00  = 500.00

Subtotal: 2000.00
GST 18%: 360.00
Total: INR 2360.00
```

Output:

```json
{
  "invoice_number": "INV-2041",
  "invoice_date": "2025-10-12",
  "vendor_name": "Mapusa Electronics",
  "customer_name": "Rahul Naik",
  "currency": "INR",
  "line_items": [
    {
      "description": "Laptop charger",
      "quantity": 1.0,
      "unit_price": 1500.0,
      "total": 1500.0
    },
    {
      "description": "USB cable",
      "quantity": 2.0,
      "unit_price": 250.0,
      "total": 500.0
    }
  ],
  "subtotal": 2000.0,
  "tax": 360.0,
  "total": 2360.0
}
```

**Project structure**

```
task1.py            # extractor script
requirements.txt    # dependencies
sample_invoice.txt  # example input
README.md
```

**Tech stack**

Python 3, Pydantic, Requests, OpenRouter API.

# Task-2

# Document Q&A Assistant with RAG & Vector Embeddings

A Retrieval-Augmented Generation (RAG) system that answers questions **strictly from your own PDF/text documents** and cites the source page for every answer.

## Features
- Loads PDF and text documents (`.pdf`, `.txt`, `.md`)
- Recursive text chunking with overlap
- Vector embeddings using a free local model (`all-MiniLM-L6-v2`)
- ChromaDB vector store, persisted locally (no database server needed)
- `RetrievalQA` chain with a prompt that forbids outside knowledge
- Source citations with file name and page number
- Refuses to answer when the information isn't in the documents

## Tech Stack
Python 3, LangChain, ChromaDB, HuggingFace Sentence Transformers, OpenRouter (LLM)

## How It Works
1. **Load**: `PyPDFLoader` reads each PDF page as a document.
2. **Chunk**: `RecursiveCharacterTextSplitter` splits text into 1000-character chunks with 200 overlap, keeping page metadata.
3. **Embed**: each chunk is converted into a vector by `all-MiniLM-L6-v2`.
4. **Store**: vectors are saved in ChromaDB (`./chroma_db`).
5. **Retrieve**: the question is embedded and the most similar chunks are fetched.
6. **Answer**: the LLM answers using only the retrieved context.
7. **Cite**: the source file and page number are printed with the answer.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt

# Windows PowerShell
$env:OPENROUTER_API_KEY="your_key_here"

# Mac/Linux
export OPENROUTER_API_KEY=your_key_here
```

**3. Add documents** to the `docs/` folder.

## Usage
```bash
# Ask a single question
python task2.py --docs ./docs --ask "How many days of paid annual leave do employees get?"

# Interactive mode (type exit to quit)
python task2.py --docs ./docs

# Re-index after adding or changing documents
python task2.py --docs ./docs --rebuild
```

## Example Output
```
Answer:
24 days of paid annual leave per calendar year.

Sources:
  - novabyte_handbook.pdf, page 3
```

Out-of-scope question:
```
Answer:
I couldn't find that in the provided documents.
```

## Sample Questions (for the included handbook)
| Question | Expected answer | Page |
|---|---|---|
| How many days of paid annual leave do employees get? | 24 days | 3 |
| How long must passwords be? | At least 14 characters | 5 |
| How much is the annual learning budget? | Rs 40,000 | 6 |

# Task-3
