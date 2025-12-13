"""Evaluation framework for Laws of Form benchmarks."""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

# =============================================================================
# Pricing
# =============================================================================

PRICING = {  # USD per 1M tokens
    # Anthropic
    "claude-opus-4-5-20251101": {"input": 5.0, "output": 25.0},
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    # OpenAI
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    # Google
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    """Calculate cost based on model pricing."""
    prices = PRICING.get(model, {"input": 0, "output": 0})
    prompt_cost = (prompt_tokens / 1_000_000) * prices["input"]
    completion_cost = (completion_tokens / 1_000_000) * prices["output"]
    return {
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": prompt_cost + completion_cost,
    }


# =============================================================================
# Prompts
# =============================================================================

PROMPT_TEMPLATE = """You are an expert in evaluating Laws of Form expressions. \
Your task is to analyze an expression that represents a structure of distinctions \
and reduce it to its simplest form using two fundamental axioms.

Here is the expression you need to evaluate:

<expression>
{expression}
</expression>

## The form and axioms

#### Axiom 1. The law of calling
Multiple adjacent boundaries with nothing else inside them condense into one.
Example: ()() = ()

#### Axiom 2. The law of crossing
Two nested boundaries with nothing else between them annihilate to void.
Example: (()) = void

## Instructions

1. Identify the structure
2. Look for opportunities to apply axioms (I1 or I2)
3. Apply reductions iteratively until no more reductions are possible
4. State whether the final result is marked (reduces to ()) or unmarked (reduces to void)

After completing your reduction, provide your final answer in this exact format:

<answer>X</answer>

where X is either:
- unmarked (if the expression reduces to void)
- marked (if structure remains)
"""

COMPOSITE_PROMPT_TEMPLATE = """You are an expert in evaluating Laws of Form expressions.
Your task is to analyze an expression that represents a structure of distinctions \
and reduce it to its simplest form using two fundamental axioms.

Here are the expressions you need to evaluate:

<expressions>
{expressions}
</expressions>

#### Axiom 1. The law of calling
Multiple adjacent boundaries condense into one: ()() = ()

#### Axiom 2. The law of crossing
Two nested boundaries annihilate to void: (()) = void

## Instructions

1. Identify the structure
2. Look for opportunities to apply axioms (I1 or I2)
3. Apply reductions iteratively until no more reductions are possible
4. State whether the final result is marked (reduces to ()) or unmarked (reduces to void)

After working through ALL expressions, provide your final answers in this exact JSON format:

```json
{{"E1": "marked", "E2": "unmarked", ..., "total_marked": N}}
```

Where each E# is either "marked" or "unmarked", and total_marked is the count of marked expressions.
"""


# =============================================================================
# Answer Extraction
# =============================================================================

def extract_answer(response: str | None) -> str:
    """Extract the final answer from LLM response. Returns 'marked', 'unmarked', or 'unknown'."""
    if response is None:
        return "unknown"

    pattern = r'<answer>\s*(marked|unmarked|\(\)|void)\s*</answer>'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        ans = match.group(1).lower()
        return "marked" if ans in ("marked", "()") else "unmarked"

    response_lower = response.lower()
    mark_pos = max(response_lower.rfind("()"), response_lower.rfind("marked"))
    void_pos = max(response_lower.rfind("void"), response_lower.rfind("unmarked"))

    if mark_pos > void_pos:
        return "marked"
    elif void_pos > mark_pos:
        return "unmarked"
    return "unknown"


def extract_composite_answer(response: str | None, n: int) -> dict:
    """
    Extract composite answer from LLM response.

    Returns dict with:
        - "items": list of "marked"/"unmarked"/"unknown" for each item
        - "total_marked": int or -1 if unparseable
    """
    empty = {"items": ["unknown"] * n, "total_marked": -1}
    if response is None:
        return empty

    # Try to find JSON in the response
    # Look for ```json ... ``` or just { ... }
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    if not json_match:
        json_match = re.search(r'(\{[^{}]*"E1"[^{}]*\})', response, re.DOTALL)

    if json_match:
        try:
            data = json.loads(json_match.group(1))
            items = []
            for i in range(1, n + 1):
                key = f"E{i}"
                val = data.get(key, "unknown")
                if isinstance(val, str):
                    val = val.lower()
                    if val in ("marked", "()"):
                        items.append("marked")
                    elif val in ("unmarked", "void"):
                        items.append("unmarked")
                    else:
                        items.append("unknown")
                else:
                    items.append("unknown")
            total = data.get("total_marked", -1)
            if not isinstance(total, int) or total < 0 or total > n:
                total = -1
            return {"items": items, "total_marked": total}
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: try to extract total_marked from text
    match = re.search(r'total_marked["\s:]+(\d+)', response, re.IGNORECASE)
    if match:
        total = int(match.group(1))
        if 0 <= total <= n:
            return {"items": ["unknown"] * n, "total_marked": total}

    return empty


# =============================================================================
# Task Types
# =============================================================================

class SingleTask:
    """Single expression evaluation task."""

    @staticmethod
    def format_prompt(case: dict) -> str:
        return PROMPT_TEMPLATE.format(expression=case["input"])

    @staticmethod
    def extract(response: str, case: dict) -> str:
        return extract_answer(response)

    @staticmethod
    def is_correct(answer: Any, case: dict) -> bool:
        return answer == case["target"]

    @staticmethod
    def make_result(
        case: dict, provider: str, model: str, response: str, answer: Any, metadata: dict
    ) -> dict:
        # Calculate cost from usage
        usage = metadata.get("usage", {})
        cost = calculate_cost(
            model,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
        return {
            "task_id": case["id"],
            "answer": answer,
            "correct": SingleTask.is_correct(answer, case),
            "target": case["target"],
            "input": case["input"],
            "difficulty": case["difficulty"],
            "metadata": {
                "model": model,
                "provider": provider,
                "start_timestamp": metadata.get("start_timestamp"),
                "end_timestamp": metadata.get("end_timestamp"),
                "response": response,
                "kwargs": metadata.get("kwargs", {}),
                "usage": usage,
                "cost": cost,
            },
        }


class CompositeTask:
    """Composite (multi-expression) evaluation task."""

    @staticmethod
    def format_prompt(case: dict) -> str:
        lines = [f"E{i}. {expr}\n\n" for i, expr in enumerate(case["expressions"], 1)]
        return COMPOSITE_PROMPT_TEMPLATE.format(expressions="\n".join(lines))

    @staticmethod
    def extract(response: str, case: dict) -> dict:
        return extract_composite_answer(response, case["group_size"])

    @staticmethod
    def make_result(
        case: dict, provider: str, model: str, response: str, answer: dict, metadata: dict
    ) -> dict:
        target_items = case["targets"]  # Already "marked"/"unmarked"
        extracted_items = answer["items"]

        # Compute per-item correctness
        per_item_correct = [
            ext == tgt for ext, tgt in zip(extracted_items, target_items)
        ]

        # Calculate cost from usage
        usage = metadata.get("usage", {})
        cost = calculate_cost(
            model,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

        return {
            "task_id": case["id"],
            "answer": extracted_items,
            "answer_count": answer["total_marked"],
            "correct": all(per_item_correct),
            "count_correct": answer["total_marked"] == case["count"],
            "target": target_items,
            "target_count": case["count"],
            "expressions": case["expressions"],
            "difficulty": case["difficulty"],
            "per_item_correct": per_item_correct,
            "metadata": {
                "model": model,
                "provider": provider,
                "start_timestamp": metadata.get("start_timestamp"),
                "end_timestamp": metadata.get("end_timestamp"),
                "response": response,
                "kwargs": metadata.get("kwargs", {}),
                "usage": usage,
                "cost": cost,
            },
        }


# =============================================================================
# Live Evaluation (async, for quick_test)
# =============================================================================

async def eval_openai_live(prompt: str, model: str) -> tuple[str, dict]:
    """Live OpenAI API call. Returns (response_text, metadata)."""
    from openai import AsyncOpenAI
    kwargs = {"temperature": 1}
    async with AsyncOpenAI() as client:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
    text = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    return text, {"kwargs": kwargs, "usage": usage}


async def eval_anthropic_live(prompt: str, model: str) -> tuple[str, dict]:
    """Live Anthropic API call. Returns (response_text, metadata)."""
    from anthropic import AsyncAnthropic
    kwargs = {"max_tokens": 60000}
    client = AsyncAnthropic()
    async with client.messages.stream(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **kwargs
    ) as stream:
        message = await stream.get_final_message()
    text = message.content[0].text
    usage = {
        "prompt_tokens": message.usage.input_tokens,
        "completion_tokens": message.usage.output_tokens,
    }
    return text, {"kwargs": kwargs, "usage": usage}


async def eval_google_live(prompt: str, model: str) -> tuple[str, dict]:
    """Live Google API call. Returns (response_text, metadata)."""
    from google import genai
    kwargs = {}
    async with genai.Client().aio as client:
        response = await client.models.generate_content(model=model, contents=prompt)
    text = response.text
    usage = {
        "prompt_tokens": response.usage_metadata.prompt_token_count,
        "completion_tokens": response.usage_metadata.candidates_token_count,
    }
    return text, {"kwargs": kwargs, "usage": usage}


LIVE_EVAL_FNS = {
    "openai": eval_openai_live,
    "anthropic": eval_anthropic_live,
    "google": eval_google_live,
}


# =============================================================================
# Batch Evaluation
# =============================================================================

def log_batch_event(log_file: str, event: str, provider: str, batch_id: str, **data):
    """Append a batch event to the log file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        "provider": provider,
        "batch_id": batch_id,
        **data
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def create_openai_batch(cases: list, task_type: type, model: str) -> str:
    """Create OpenAI batch job. Returns batch ID."""
    from openai import OpenAI
    client = OpenAI()
    jsonl_content = ""
    for case in cases:
        prompt = task_type.format_prompt(case)
        request = {
            "custom_id": case["id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1,
            }
        }
        jsonl_content += json.dumps(request) + "\n"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(jsonl_content)
        temp_path = f.name

    with open(temp_path, 'rb') as f:
        file_obj = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    return batch.id


def create_anthropic_batch(cases: list, task_type: type, model: str) -> str:
    """Create Anthropic batch job. Returns batch ID."""
    from anthropic import Anthropic
    client = Anthropic()
    requests_list = []
    for case in cases:
        prompt = task_type.format_prompt(case)
        requests_list.append({
            "custom_id": case["id"],
            "params": {
                "model": model,
                "max_tokens": 64000,
                "temperature": 1,
                "messages": [{"role": "user", "content": prompt}]
            }
        })
    batch = client.messages.batches.create(requests=requests_list)
    return batch.id


def create_google_batch(cases: list, task_type: type, model: str) -> str:
    """Create Google batch job. Returns batch name."""
    from google import genai
    client = genai.Client()
    inline_requests = []
    for case in cases:
        prompt = task_type.format_prompt(case)
        inline_requests.append({
            "contents": [{"parts": [{"text": prompt}], "role": "user"}]
        })
    batch_job = client.batches.create(
        model=f"models/{model}",
        src=inline_requests,
        config={"display_name": f"lof-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"},
    )
    return batch_job.name


def check_openai_batch(batch_id: str) -> tuple[str, list | None]:
    """Check OpenAI batch status. Returns (status, results_or_none)."""
    from openai import OpenAI
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    if batch.status == "completed":
        output_file = client.files.content(batch.output_file_id)
        results = []
        for line in output_file.text.strip().split("\n"):
            result = json.loads(line)
            results.append({
                "id": result["custom_id"],
                "response": result["response"]["body"]["choices"][0]["message"]["content"],
            })
        return "completed", results
    return batch.status, None


def check_anthropic_batch(batch_id: str) -> tuple[str, list | None]:
    """Check Anthropic batch status. Returns (status, results_or_none)."""
    from anthropic import Anthropic
    client = Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status == "ended":
        results = []
        for result in client.messages.batches.results(batch_id):
            results.append({
                "id": result.custom_id,
                "response": result.result.message.content[0].text,
            })
        return "completed", results
    return batch.processing_status, None


def check_google_batch(batch_name: str) -> tuple[str, list | None]:
    """Check Google batch status. Returns (status, results_or_none)."""
    from google import genai
    client = genai.Client()
    batch = client.batches.get(name=batch_name)
    if batch.state.name != "JOB_STATE_SUCCEEDED":
        return batch.state.name.lower(), None
    results = []
    if batch.dest and batch.dest.inlined_responses:
        for i, inline_resp in enumerate(batch.dest.inlined_responses):
            if inline_resp.response:
                text = inline_resp.response.text
            else:
                text = f"ERROR: {inline_resp.error}"
            results.append({"id": f"idx_{i}", "response": text})
    return "completed", results


BATCH_CREATE_FNS = {
    "openai": create_openai_batch,
    "anthropic": create_anthropic_batch,
    "google": create_google_batch,
}

BATCH_CHECK_FNS = {
    "openai": check_openai_batch,
    "anthropic": check_anthropic_batch,
    "google": check_google_batch,
}


# =============================================================================
# Main Functions
# =============================================================================

def quick_test(
    cases: list,
    task_type: type = SingleTask,
    n: int | None = None,
    providers: list[str] | None = None,
    models: dict[str, str] | None = None,
) -> list[dict]:
    """
    Run live API calls for quick testing.

    Args:
        cases: Test cases to evaluate
        task_type: SingleTask or CompositeTask
        n: Number of cases to test (default: all)
        providers: List of providers to use (default: ["openai", "anthropic", "google"])
        models: Dict mapping provider -> model name

    Returns:
        List of result dicts
    """
    if providers is None:
        providers = ["openai", "anthropic", "google"]
    if models is None:
        models = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
            "google": "gemini-2.0-flash",
        }
    if n is not None:
        cases = cases[:n]

    print(f"Running {task_type.__name__} on {len(cases)} cases (live)...")

    async def _run_single(case, provider, model, fn, prompt):
        """Run a single eval with timestamps."""
        start = datetime.now().isoformat()
        try:
            response, api_meta = await fn(prompt, model)
            end = datetime.now().isoformat()
            answer = task_type.extract(response, case)
            metadata = {
                "start_timestamp": start,
                "end_timestamp": end,
                **api_meta,
            }
        except Exception as e:
            end = datetime.now().isoformat()
            response = f"ERROR: {e}"
            if task_type == CompositeTask:
                answer = {"items": ["unknown"] * case["group_size"], "total_marked": -1}
            else:
                answer = "unknown"
            metadata = {"start_timestamp": start, "end_timestamp": end, "kwargs": {}, "usage": {}}
        return task_type.make_result(case, provider, model, response, answer, metadata)

    async def _run():
        tasks = []
        for case in cases:
            prompt = task_type.format_prompt(case)
            for provider in providers:
                if provider not in LIVE_EVAL_FNS:
                    continue
                model = models.get(provider)
                if not model:
                    continue
                fn = LIVE_EVAL_FNS[provider]
                tasks.append(_run_single(case, provider, model, fn, prompt))
        return await asyncio.gather(*tasks)

    results = asyncio.run(_run())

    if task_type == SingleTask:
        correct = sum(1 for r in results if r["correct"])
        print(f"\nResults: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
        for r in results:
            status = "✓" if r["correct"] else "✗"
            inp = r["input"][:25]
            prov = r["metadata"]["provider"]
            print(f"  {status} {r['task_id']} ({prov}): {inp}... → {r['answer']}")
    else:
        # Composite metrics
        n = len(results)
        all_correct = sum(1 for r in results if r["correct"])
        count_correct = sum(1 for r in results if r["count_correct"])
        per_item_accs = [sum(r["per_item_correct"]) / len(r["per_item_correct"]) for r in results]
        avg_item_acc = sum(per_item_accs) / n if n > 0 else 0
        print(f"\nResults ({n} cases):")
        print(f"  Per-item accuracy: {100*avg_item_acc:.1f}%")
        print(f"  All-correct@8:     {all_correct}/{n} ({100*all_correct/n:.1f}%)")
        print(f"  Count exact match: {count_correct}/{n} ({100*count_correct/n:.1f}%)")
        for r in results:
            status = "✓" if r["correct"] else "✗"
            items = sum(r["per_item_correct"])
            total = len(r["target"])
            cnt = r["answer_count"]
            tgt = r["target_count"]
            prov = r["metadata"]["provider"]
            tid = r["task_id"]
            print(f"  {status} {tid} ({prov}): {items}/{total} items, count={cnt} (tgt {tgt})")
    return results


def run_eval(
    cases: list,
    task_type: type = SingleTask,
    providers: list[str] | None = None,
    models: dict[str, str] | None = None,
    poll_interval: int = 60,
    batch_log: str = "batch_log.jsonl",
) -> list[dict]:
    """
    Run batch API calls. Submits jobs, polls until complete, returns results.

    Args:
        cases: Test cases to evaluate
        task_type: SingleTask or CompositeTask
        providers: List of providers to use
        models: Dict mapping provider -> model name
        poll_interval: Seconds between status checks
        batch_log: Path to batch log file

    Returns:
        List of result dicts
    """
    if providers is None:
        providers = ["openai", "anthropic", "google"]
    if models is None:
        models = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
            "google": "gemini-2.0-flash",
        }

    print(f"Submitting batch jobs for {len(cases)} cases...")
    batch_ids = {}

    for provider in providers:
        if provider not in BATCH_CREATE_FNS:
            continue
        model = models.get(provider)
        if not model:
            continue
        try:
            batch_id = BATCH_CREATE_FNS[provider](cases, task_type, model)
            batch_ids[provider] = batch_id
            log_batch_event(
                batch_log, "submitted", provider, batch_id, model=model, n_cases=len(cases)
            )
            print(f"  {provider}: {batch_id}")
        except Exception as e:
            print(f"  {provider}: FAILED - {e}")

    print(f"\nPolling for completion (every {poll_interval}s)...")
    completed = {}
    while len(completed) < len(batch_ids):
        time.sleep(poll_interval)
        for provider, batch_id in batch_ids.items():
            if provider in completed:
                continue
            try:
                status, results = BATCH_CHECK_FNS[provider](batch_id)
                if results:
                    completed[provider] = results
                    log_batch_event(batch_log, "completed", provider, batch_id)
                    print(f"  {provider}: completed ({len(results)} results)")
                else:
                    print(f"  {provider}: {status}")
            except Exception as e:
                print(f"  {provider}: error - {e}")

    # Build final results
    case_lookup = {c["id"]: c for c in cases}
    all_results = []
    for provider, batch_results in completed.items():
        model = models[provider]
        for r in batch_results:
            case = case_lookup.get(r["id"])
            if not case:
                continue
            response = r["response"]
            answer = task_type.extract(response, case)
            all_results.append(task_type.make_result(case, provider, model, response, answer))

    return all_results


def fetch_batch_results(
    cases: list,
    task_type: type = SingleTask,
    models: dict[str, str] | None = None,
    batch_log: str = "batch_log.jsonl",
) -> list[dict]:
    """
    Fetch results from completed batch jobs (re-download if needed).

    Args:
        cases: Test cases (needed to process results)
        task_type: SingleTask or CompositeTask
        models: Dict mapping provider -> model name
        batch_log: Path to batch log file

    Returns:
        List of processed result dicts
    """
    if models is None:
        models = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
            "google": "gemini-2.0-flash",
        }

    # Load all submitted batch IDs (including completed ones)
    batch_ids = {}
    with open(batch_log) as f:
        for line in f:
            entry = json.loads(line)
            if entry["event"] == "submitted":
                batch_ids[entry["provider"]] = entry["batch_id"]

    if not batch_ids:
        print("No batches found in log")
        return []

    print(f"Fetching results for: {list(batch_ids.keys())}")
    completed = {}
    for provider, batch_id in batch_ids.items():
        try:
            status, results = BATCH_CHECK_FNS[provider](batch_id)
            if results:
                completed[provider] = results
                print(f"  {provider}: fetched {len(results)} results")
            else:
                print(f"  {provider}: {status} (no results)")
        except Exception as e:
            print(f"  {provider}: error - {e}")

    # Build final results
    case_lookup = {c["id"]: c for c in cases}
    all_results = []
    for provider, batch_results in completed.items():
        model = models[provider]
        for r in batch_results:
            case = case_lookup.get(r["id"])
            if not case:
                continue
            response = r["response"]
            answer = task_type.extract(response, case)
            metadata = {"kwargs": {}, "usage": {}}
            all_results.append(task_type.make_result(
                case, provider, model, response, answer, metadata
            ))

    return all_results


def resume_batch(
    cases: list,
    task_type: type = SingleTask,
    models: dict[str, str] | None = None,
    batch_log: str = "batch_log.jsonl",
    poll_interval: int = 60,
    skip_providers: list[str] | None = None,
) -> list[dict]:
    """
    Resume polling for pending batch jobs and process results.

    Args:
        cases: Test cases (needed to process results)
        task_type: SingleTask or CompositeTask
        models: Dict mapping provider -> model name
        batch_log: Path to batch log file
        poll_interval: Seconds between status checks
        skip_providers: List of providers to skip (e.g., cancelled batches)

    Returns:
        List of processed result dicts (same format as run_eval)
    """
    if models is None:
        models = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
            "google": "gemini-2.0-flash",
        }

    # Load batch IDs (latest submission per provider, excluding completed)
    batch_ids = {}
    with open(batch_log) as f:
        for line in f:
            entry = json.loads(line)
            if entry["event"] == "submitted":
                batch_ids[entry["provider"]] = entry["batch_id"]
            elif entry["event"] in ("completed", "cancelled"):
                batch_ids.pop(entry["provider"], None)

    # Skip specified providers
    if skip_providers:
        for p in skip_providers:
            batch_ids.pop(p, None)

    # Poll for pending batches
    if not batch_ids:
        print("No pending batches, fetching completed results...")
    else:
        print(f"Polling pending batches: {list(batch_ids.keys())}")
        failed = set()
        while len(failed) < len(batch_ids):
            time.sleep(poll_interval)
            all_done = True
            for provider, batch_id in batch_ids.items():
                if provider in failed:
                    continue
                try:
                    status, results = BATCH_CHECK_FNS[provider](batch_id)
                    if results:
                        log_batch_event(batch_log, "completed", provider, batch_id)
                        print(f"  {provider}: completed ({len(results)} results)")
                    elif status in ("failed", "cancelled", "expired"):
                        failed.add(provider)
                        log_batch_event(batch_log, "cancelled", provider, batch_id)
                        print(f"  {provider}: {status} (skipping)")
                        continue
                    else:
                        print(f"  {provider}: {status}")
                        all_done = False
                except Exception as e:
                    failed.add(provider)
                    print(f"  {provider}: error - {e} (skipping)")
            if all_done:
                break

    # Fetch all results (use fetch_batch_results)
    return fetch_batch_results(cases, task_type, models, batch_log)


def analyze(results: list[dict]) -> dict:
    """
    Analyze evaluation results.

    For SingleTask: returns accuracy by model and difficulty.
    For CompositeTask: returns three metrics by model and difficulty:
        - per_item_accuracy: average per-item accuracy
        - all_correct_rate: fraction with all items correct
        - count_match_rate: fraction with exact count match
    """
    if not results:
        return {"by_model": {}, "by_difficulty": {}}

    # Detect composite by presence of per_item_correct
    is_composite = "per_item_correct" in results[0]

    if not is_composite:
        # SingleTask analysis
        by_model = defaultdict(lambda: {"correct": 0, "total": 0})
        by_diff = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

        for r in results:
            model = r["metadata"]["model"]
            by_model[model]["total"] += 1
            if r["correct"]:
                by_model[model]["correct"] += 1
            by_diff[model][r["difficulty"]]["total"] += 1
            if r["correct"]:
                by_diff[model][r["difficulty"]]["correct"] += 1

        return {
            "by_model": {m: {
                "accuracy": d["correct"]/d["total"] if d["total"] > 0 else 0,
                "correct": d["correct"],
                "total": d["total"],
            } for m, d in by_model.items()},
            "by_difficulty": {m: {diff: {
                "accuracy": d["correct"]/d["total"] if d["total"] > 0 else 0,
            } for diff, d in diffs.items()} for m, diffs in by_diff.items()},
        }

    # CompositeTask analysis - three metrics
    by_model = defaultdict(lambda: {
        "total": 0,
        "item_acc_sum": 0.0,
        "all_correct": 0,
        "count_correct": 0,
    })
    by_diff = defaultdict(lambda: defaultdict(lambda: {
        "total": 0,
        "item_acc_sum": 0.0,
        "all_correct": 0,
        "count_correct": 0,
    }))

    for r in results:
        model = r["metadata"]["model"]
        diff = r["difficulty"]
        # Calculate per-item accuracy from per_item_correct
        item_acc = sum(r["per_item_correct"]) / len(r["per_item_correct"])

        by_model[model]["total"] += 1
        by_model[model]["item_acc_sum"] += item_acc
        if r["correct"]:
            by_model[model]["all_correct"] += 1
        if r["count_correct"]:
            by_model[model]["count_correct"] += 1

        by_diff[model][diff]["total"] += 1
        by_diff[model][diff]["item_acc_sum"] += item_acc
        if r["correct"]:
            by_diff[model][diff]["all_correct"] += 1
        if r["count_correct"]:
            by_diff[model][diff]["count_correct"] += 1

    def metrics(d):
        t = d["total"]
        if t == 0:
            return {
                "accuracy": 0, "correct": 0, "total": 0,
                "per_item_accuracy": 0, "all_correct_rate": 0, "count_match_rate": 0,
            }
        return {
            # Standard keys (accuracy = per_item, correct = all_correct count)
            "accuracy": d["item_acc_sum"] / t,
            "correct": d["all_correct"],
            "total": t,
            # Composite-specific keys
            "per_item_accuracy": d["item_acc_sum"] / t,
            "all_correct_rate": d["all_correct"] / t,
            "count_match_rate": d["count_correct"] / t,
        }

    return {
        "by_model": {m: metrics(d) for m, d in by_model.items()},
        "by_difficulty": {
            m: {diff: metrics(d) for diff, d in diffs.items()}
            for m, diffs in by_diff.items()
        },
    }
