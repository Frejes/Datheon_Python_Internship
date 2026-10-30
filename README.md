# Task-1
# Structured LLM Prompt Pipeline & JSON Extractor

A Python script that turns unstructured text documents (invoices and resumes) into **strictly validated JSON** using an LLM API.

## How it works

1. **Schemas (Pydantic):** `Invoice` and `Resume` models define exactly what valid output looks like. The invoice model also checks that `subtotal + tax == total`.
2. **Parameterized prompt templates:** each document type has a system instruction, a few-shot example, and a template that inserts the raw document text. Adding a new document type only requires a new entry in the `PROMPTS` dict.
3. **LLM call (OpenRouter):** the script sends the prompt to a chat model through the OpenRouter API, with the JSON Schema included in the system message and `temperature=0` for consistent output.
4. **Validation and retry loop:** the reply is parsed with Pydantic. If validation fails, the error is fed back to the model and it retries (up to 3 times).
5. **Resilience:** HTTP 429/5xx errors and network timeouts are retried with increasing delays.

## Setup

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

## Usage

```bash
python task1.py invoice sample_invoice.txt
python task1.py resume  sample_resume.txt
```

Save the output to a file:

```bash
python task1.py invoice sample_invoice.txt > output_invoice.json
```

## Example

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

## Project structure

```
task1.py            # extractor script
requirements.txt    # dependencies
sample_invoice.txt  # example input
README.md
```

## Troubleshooting

| Error | Likely cause |
|---|---|
| `KeyError: 'OPENROUTER_API_KEY'` | Key not set in the current terminal window |
| `401` | Wrong or mistyped API key |
| `402` | Model needs credits; use a `:free` model or add credit |
| `404` / "No endpoints found" | Model ID is wrong or the model was removed |
| `429` / `503` | Model is busy or rate-limited; the script retries automatically, or try another model |
| Validation failed after 3 attempts | The model is not following the schema well; try a different model |

## Tech stack

Python 3, Pydantic, Requests, OpenRouter API.

# Task-2
