
import argparse
import json
import os
import sys
import time
from datetime import date
from typing import Optional

import requests
from pydantic import BaseModel, Field, ValidationError, model_validator

MAX_RETRIES = 3
RETRY_DELAYS_SECONDS = (5, 15, 30, 60)
MAX_REQUEST_RETRIES = len(RETRY_DELAYS_SECONDS) + 1
MAX_SERVER_RETRY_AFTER_SECONDS = 120
REQUEST_TIMEOUT_SECONDS = 90
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class LineItem(BaseModel):
    description: str
    quantity: float = Field(ge=0)
    unit_price: float = Field(ge=0)
    total: float = Field(ge=0)


class Invoice(BaseModel):
    invoice_number: str
    invoice_date: date = Field(description="ISO format YYYY-MM-DD")
    vendor_name: str
    customer_name: Optional[str] = None
    currency: str = Field(description="ISO 4217 code, e.g. INR, USD, CHF")
    line_items: list[LineItem]
    subtotal: float
    tax: float = 0
    total: float

    @model_validator(mode="after")
    def totals_must_add_up(self):
        for item in self.line_items:
            if abs(item.quantity * item.unit_price - item.total) > 0.01:
                raise ValueError(
                    f"line item total must equal quantity * unit_price: {item.description}"
                )
        if abs(sum(item.total for item in self.line_items) - self.subtotal) > 0.01:
            raise ValueError("line item totals must add up to subtotal")
        if abs(self.subtotal + self.tax - self.total) > 0.01:
            raise ValueError("subtotal + tax must equal total")
        return self


class Experience(BaseModel):
    company: str
    title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = Field(None, description="null if current role")
    summary: Optional[str] = None


class Resume(BaseModel):
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: list[str]
    experience: list[Experience]
    education: list[str]


PROMPTS = {
    "invoice": {
        "schema": Invoice,
        "system": (
            "You are a precise data-extraction engine. Extract invoice data from the "
            "document. Use null for missing optional fields. Never invent values. "
            "Return only JSON matching the schema."
        ),
        "examples": [
            (
                "INVOICE #A-102 | Date: 5 March 2025\nFrom: Goa Supplies Pvt Ltd\n"
                "Bill to: Sunrise Cafe\n2 x Coffee beans 1kg @ 800.00 = 1600.00\n"
                "Subtotal 1600.00  GST 18% 288.00  Total INR 1888.00",
                {
                    "invoice_number": "A-102",
                    "invoice_date": "2025-03-05",
                    "vendor_name": "Goa Supplies Pvt Ltd",
                    "customer_name": "Sunrise Cafe",
                    "currency": "INR",
                    "line_items": [
                        {"description": "Coffee beans 1kg", "quantity": 2,
                         "unit_price": 800.0, "total": 1600.0}
                    ],
                    "subtotal": 1600.0,
                    "tax": 288.0,
                    "total": 1888.0,
                },
            )
        ],
    },
    "resume": {
        "schema": Resume,
        "system": (
            "You are a precise resume parser. Extract the candidate's details. "
            "Use null for missing optional fields. Never invent values. "
            "Return only JSON matching the schema."
        ),
        "examples": [
            (
                "Asha Rao | asha@mail.com | +91 98765 43210\n"
                "Skills: Python, SQL\n"
                "Data Analyst, Infosys, Jun 2022 - Present: built dashboards.\n"
                "B.E. Computer Engineering, Goa University, 2022",
                {
                    "full_name": "Asha Rao",
                    "email": "asha@mail.com",
                    "phone": "+91 98765 43210",
                    "skills": ["Python", "SQL"],
                    "experience": [
                        {"company": "Infosys", "title": "Data Analyst",
                         "start_date": "2022-06", "end_date": None,
                         "summary": "Built dashboards"}
                    ],
                    "education": ["B.E. Computer Engineering, Goa University, 2022"],
                },
            )
        ],
    },
}


def build_prompt(kind: str, document: str, error_feedback: str = "") -> str:
    cfg = PROMPTS[kind]
    shots = "\n\n".join(
        f"### Example input\n{inp}\n### Example output\n{json.dumps(out, indent=2)}"
        for inp, out in cfg["examples"]
    )
    prompt = f"{shots}\n\n### Input\n{document}\n### Output\n"
    if error_feedback:
        prompt += f"\nYour previous output failed validation:\n{error_feedback}\nFix it.\n"
    return prompt


def call_openrouter(system: str, prompt: str, schema: type[BaseModel]) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Set it in this terminal before running the script."
        )
    if not model:
        raise RuntimeError(
            "OPENROUTER_MODEL is not set. Set it to a model available to your OpenRouter account."
        )

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{system}\nReturn ONLY JSON matching this JSON Schema:\n"
                        f"{json.dumps(schema.model_json_schema())}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise RuntimeError("OpenRouter returned an empty or non-text response.")
    return (
        content.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )


def _http_error_detail(response: Optional[requests.Response]) -> str:
    if response is None:
        return "No response body was provided."

    try:
        body = response.json()
    except requests.exceptions.JSONDecodeError:
        return response.text[:500] or "No response body was provided."

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
    return response.text[:500] or "No error details were provided."


def extract(kind: str, document: str) -> BaseModel:
    cfg = PROMPTS[kind]
    schema = cfg["schema"]
    feedback = ""

    for attempt in range(1, MAX_RETRIES + 1):
        print(
            f"[attempt {attempt}/{MAX_RETRIES}] Sending request to OpenRouter...",
            file=sys.stderr,
            flush=True,
        )
        for request_attempt in range(1, MAX_REQUEST_RETRIES + 1):
            try:
                response_text = call_openrouter(
                    cfg["system"], build_prompt(kind, document, feedback), schema
                )
                break
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else None
                if status_code not in RETRYABLE_STATUS_CODES:
                    raise
                detail = _http_error_detail(e.response)
                if request_attempt == MAX_REQUEST_RETRIES:
                    raise RuntimeError(
                        f"OpenRouter rejected the request after {MAX_REQUEST_RETRIES} "
                        f"attempts (HTTP {status_code}): {detail}\n"
                        "For HTTP 429, this is usually a rate/quota limit. Wait for "
                        "the limit to reset, check your OpenRouter account limits or "
                        "credits, or try a different available model."
                    ) from e

                retry_after = (
                    e.response.headers.get("Retry-After")
                    if e.response is not None
                    else None
                )
                try:
                    server_delay = float(retry_after) if retry_after else None
                except ValueError:
                    server_delay = None

                if server_delay is not None and server_delay > MAX_SERVER_RETRY_AFTER_SECONDS:
                    raise RuntimeError(
                        f"OpenRouter returned HTTP {status_code} and requested a "
                        f"{server_delay:g}-second wait: {detail}\n"
                        "Wait that long before retrying, or check your OpenRouter "
                        "account limits/credits and try a different available model."
                    ) from e

                delay = (
                    server_delay
                    if server_delay is not None
                    else RETRY_DELAYS_SECONDS[request_attempt - 1]
                )
                print(
                    f"OpenRouter returned HTTP {status_code}; retrying in "
                    f"{delay:g} seconds...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                if request_attempt == MAX_REQUEST_RETRIES:
                    raise RuntimeError(
                        f"Could not connect to OpenRouter after "
                        f"{MAX_REQUEST_RETRIES} request attempts. Check your "
                        "internet connection and try again."
                    ) from e

                delay = RETRY_DELAYS_SECONDS[request_attempt - 1]
                print(
                    f"OpenRouter connection failed; retrying in {delay} seconds...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)

        try:
            return schema.model_validate_json(response_text)
        except ValidationError as e:
            feedback = str(e)
            print(f"[attempt {attempt}] validation failed, retrying...", file=sys.stderr)

    raise RuntimeError(f"Could not get valid output after {MAX_RETRIES} attempts:\n{feedback}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=PROMPTS.keys())
    ap.add_argument("file", help="path to a .txt file with the raw document")
    args = ap.parse_args()

    with open(args.file, encoding="utf-8") as f:
        text = f.read()

    result = extract(args.kind, text)
    print(result.model_dump_json(indent=2))
