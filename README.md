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

Create an API key at [openrouter.ai](https://openrouter.ai) and set it as an environment variable. 

Windows PowerShell:
```powershell
$env:OPENROUTER_API_KEY="YOUR_KEY"
$env:OPENROUTER_MODEL="recursive/flint-v1.0:free"
```

Mac/Linux:
```bash
export OPENROUTER_API_KEY="YOUR_KEY"
export OPENROUTER_MODEL="recursive/flint-v1.0:free"
```

`OPENROUTER_MODEL` is optional. Any chat model ID from [openrouter.ai/models](https://openrouter.ai/models) works.

**Usage**

```bash
python task1.py invoice sample_invoice.txt
python task1.py resume  sample_resume.txt
```

Save the output to a file:

```bash
python task1.py invoice sample_invoice.txt > output_invoice.json
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

# Task 2: Document Q&A Assistant with RAG & Vector Embeddings

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

## Project Structure
```
task2/
├── task2.py            # RAG pipeline
├── requirements.txt    # dependencies
├── README.md
├── docs/               # put your PDFs here
│   └── novabyte_handbook.pdf   # sample document
└── chroma_db/          # auto-created vector store
```

## How It Works
1. **Load**: `PyPDFLoader` reads each PDF page as a document.
2. **Chunk**: `RecursiveCharacterTextSplitter` splits text into 1000-character chunks with 200 overlap, keeping page metadata.
3. **Embed**: each chunk is converted into a vector by `all-MiniLM-L6-v2`.
4. **Store**: vectors are saved in ChromaDB (`./chroma_db`).
5. **Retrieve**: the question is embedded and the most similar chunks are fetched.
6. **Answer**: the LLM answers using only the retrieved context.
7. **Cite**: the source file and page number are printed with the answer.

## Setup

**1. Install dependencies**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

**2. Set your API key** (free key from [openrouter.ai](https://openrouter.ai))
```bash
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
| What is the daily meal allowance in Switzerland? | CHF 40 | 4 |
| How long must passwords be? | At least 14 characters | 5 |
| How much is the annual learning budget? | Rs 40,000 | 6 |
| Who is the CEO? | Meera Kamat | 2 |

## Configuration
| Setting | Location | Default |
|---|---|---|
| Chunk size / overlap | `chunk_documents()` | 1000 / 200 |
| Chunks retrieved | `build_qa_chain()` (`k`) | 4 |
| Embedding model | `EMBED_MODEL` | all-MiniLM-L6-v2 |
| Vector store folder | `PERSIST_DIR` | ./chroma_db |
| LLM | `get_llm()` | recursive/flint-v1.0:free (OpenRouter) |

## Troubleshooting
- **`No module named ...`**: activate the virtual environment and rerun `pip install -r requirements.txt`.
- **API key error**: set `OPENROUTER_API_KEY` in the same terminal window you run the script from.
- **Old answers after changing PDFs**: run once with `--rebuild`.
- **Slow first run**: the embedding model is downloaded and documents are indexed once; later runs are fast.

## Notes
- The sample handbook is fictional and included only for testing.
- Do not commit your API key. Keep it in an environment variable.
- Add `chroma_db/` and `venv/` to `.gitignore` if using Git.
