# Task-1
# Structured LLM Prompt Pipeline & JSON Extractor #

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
$env:OPENROUTER_MODEL=""
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

# Multi-Tool Autonomous AI Agent

A console AI agent built with LangChain that chooses and uses tools on its own to solve multi-step research questions. Every step of its reasoning is printed as **Thought → Action → Observation → Final Answer**.

## Features

| Tool | What it does |
|------|--------------|
| `web_search` | Real-time web search through DuckDuckGo (no API key needed) |
| `calculator` | Safe math evaluator built on Python's `ast` module (no `eval()`) |
| `read_local_file` | Reads text files from the `workspace/` folder only |

- Works with OpenRouter, Anthropic or OpenAI models (selected by whichever API key is set).
- The agent loop is written by hand, so each reasoning step is visible in the console.
- Safety limits: sandboxed file access, a calculator that rejects arbitrary code, and a maximum step count.

## How it works

```
You ask a question
        |
        v
  LLM decides  --->  Thought: what do I need next?
        |
        v
  Action: calls a tool (search / calculator / file reader)
        |
        v
  Observation: tool result is sent back to the LLM
        |
        v
  Repeat until the LLM has enough information
        |
        v
  Final Answer
```

The tools are registered with `llm.bind_tools(...)`. On each iteration the model either requests a tool call or returns a final answer. Tool results are added to the conversation as `ToolMessage`s, and the loop stops when the model asks for no more tools or `MAX_STEPS` is reached.

Python packages (`requirements.txt`):

```text
langchain-core
langchain-openai
ddgs
langchain-community
langchain-anthropic
```

`langchain-community` is only needed if you use `DuckDuckGoSearchRun`. `langchain-anthropic` is only needed for an Anthropic key.

## Setup

**1. Create and activate a virtual environment**

Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your API key** (set only one provider; the script checks them in this order)

Windows (PowerShell):
```powershell
$env:OPENROUTER_API_KEY="your-key-here"
```

Windows (Command Prompt):
```cmd
set OPENROUTER_API_KEY=your-key-here
```

macOS / Linux:
```bash
export OPENROUTER_API_KEY=your-key-here
```

The variable only lasts for the current terminal window. To use a different model, set `AGENT_MODEL` to the model name.

### Example questions

```text
What is 15% of 2480, and what is the square root of that number?
```
```text
Who won the most recent Formula 1 race, and how many points does the leader have?
```
```text
Read sales.csv, add up the amounts, then search for the current USD to INR rate and convert the total.
```

### Example output

```text
You > What is 15% of 2480, and what is the square root of that number?
[Step 1] THOUGHT: I need to compute 15% of 2480 first.
[Step 1] ACTION: calculator({'expression': '(15 * 2480) / 100'})
[Step 1] OBSERVATION: 372.0
[Step 2] THOUGHT: Now I need the square root of 372.
[Step 2] ACTION: calculator({'expression': 'sqrt(372)'})
[Step 2] OBSERVATION: 19.28730152198591

[FINAL ANSWER]
15% of 2480 is 372, and its square root is about 19.2873.
```
# Task-4

# Interactive Full-Stack AI Chatbot with Streamlit

A web-based chat application that wraps a RAG (Retrieval-Augmented Generation) pipeline in a responsive Streamlit UI. It streams answers token by token, remembers the conversation, and cites the document pages it used.

## Features

- **Streaming responses:** answers appear word by word as the model generates them.
- **Conversation memory:** the last N turns (adjustable) are sent to the model, so follow-up questions work.
- **Session state management:** chat history is stored in `st.session_state` and survives Streamlit reruns.
- **RAG over your documents:** reuses the Task 2 pipeline (ChromaDB + `all-MiniLM-L6-v2` embeddings) to answer from your PDFs/text files.
- **Source citations:** each RAG answer shows the file and page numbers it came from.
- **Strict grounding:** if the answer is not in the documents, the bot says so instead of guessing.
- **Switchable LLM provider:** OpenRouter, Anthropic, or OpenAI.
- **Sidebar controls:** model, temperature, memory length, retrieval on/off, number of chunks (k), clear chat.

## Tech Stack

| Component | Tool |
|-----------|------|
| UI | Streamlit (`st.chat_message`, `st.chat_input`, `st.write_stream`) |
| Orchestration | LangChain (prompt templates, streaming chains) |
| Vector store | ChromaDB |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | OpenRouter / Anthropic / OpenAI |
| Language | Python 3 |

## Installation

```bash
# 1. (Optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

## Configuration

Set your API key either in the app sidebar or as an environment variable:

```bash
# Windows (Command Prompt)
set OPENROUTER_API_KEY=your_key_here

# Windows (PowerShell)
$env:OPENROUTER_API_KEY="your_key_here"

# macOS / Linux
export OPENROUTER_API_KEY=your_key_here
```

Other supported variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.

## Running the App

```bash
streamlit run app.py
```

## Usage

1. Choose a provider and paste your API key in the sidebar.
2. Keep **Use retrieval (RAG)** on to answer from your documents, or turn it off for a general chatbot.
3. Type a question in the chat box.
4. Open the **Sources** dropdown under an answer to see the file and page numbers used.
5. Use **Clear conversation** to reset the memory.

## How It Works

```
User message
   -> saved in st.session_state
   -> (RAG on) search ChromaDB for the top-k chunks
        query = previous question + current question (so follow-ups still retrieve well)
   -> prompt = system instructions + retrieved context + recent history + question
   -> LLM streams the reply -> shown live with st.write_stream
   -> reply and its sources saved to st.session_state
```
