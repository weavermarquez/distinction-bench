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

## Your Approach

1. Identify the structure
2. Look for opportunities to apply axioms (I1 or I2)
3. Apply axioms iteratively until no more reductions are possible
4. State the final form

After completing your reduction, provide your final answer:

<answer>X</answer>

where X is either:
- unmarked (if the expression reduces to void)
- marked (if structure remains)
"""

COMPOSITE_PROMPT_TEMPLATE = """You are an expert in evaluating Laws of Form expressions.

Here are the expressions you need to evaluate:

<expressions>
{expressions}
</expressions>

#### Axiom 1. The law of calling
Multiple adjacent boundaries condense into one: ()() = ()

#### Axiom 2. The law of crossing
Two nested boundaries annihilate to void: (()) = void

For each expression, reduce it and determine if it's marked or unmarked.

Respond with a JSON object containing your answers. Use exactly this format:
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
    def make_result(case: dict, provider: str, model: str, response: str, answer: Any) -> dict:
        return {
            "id": case["id"],
            "input": case["input"],
            "target": case["target"],
            "difficulty": case["difficulty"],
            "provider": provider,
            "model": model,
            "response": response,
            "extracted_answer": answer,
            "correct": SingleTask.is_correct(answer, case)
        }


class CompositeTask:
    """Composite (multi-expression) evaluation task."""

    @staticmethod
    def format_prompt(case: dict) -> str:
        lines = [f"{i}. {expr}" for i, expr in enumerate(case["expressions"], 1)]
        return COMPOSITE_PROMPT_TEMPLATE.format(expressions="\n".join(lines))

    @staticmethod
    def extract(response: str, case: dict) -> dict:
        return extract_composite_answer(response, case["group_size"])

    @staticmethod
    def make_result(case: dict, provider: str, model: str, response: str, answer: dict) -> dict:
        target_items = case["targets"]  # Already "marked"/"unmarked"
        extracted_items = answer["items"]

        # Compute per-item correctness
        item_correct = [
            ext == tgt for ext, tgt in zip(extracted_items, target_items)
        ]
        n_items_correct = sum(item_correct)
        n_items = len(target_items)

        # Three metrics:
        # 1. per_item_accuracy: fraction of items correct
        # 2. all_correct: True if all items correct
        # 3. count_correct: True if total_marked matches
        return {
            "id": case["id"],
            "expressions": case["expressions"],
            "targets": target_items,
            "target_count": case["count"],
            "difficulty": case["difficulty"],
            "provider": provider,
            "model": model,
            "response": response,
            "extracted_items": extracted_items,
            "extracted_count": answer["total_marked"],
            "item_correct": item_correct,
            "n_items_correct": n_items_correct,
            "per_item_accuracy": n_items_correct / n_items if n_items > 0 else 0,
            "all_correct": n_items_correct == n_items,
            "count_correct": answer["total_marked"] == case["count"],
            # Keep "correct" for backwards compat - now means all_correct
            "correct": n_items_correct == n_items,
        }


# =============================================================================
# Live Evaluation (async, for quick_test)
# =============================================================================

async def eval_openai_live(prompt: str, model: str) -> str:
    """Live OpenAI API call."""
    from openai import AsyncOpenAI
    async with AsyncOpenAI() as client:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
    return response.choices[0].message.content


async def eval_anthropic_live(prompt: str, model: str) -> str:
    """Live Anthropic API call."""
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic()
    async with client.messages.stream(
        model=model,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        message = await stream.get_final_message()
    return message.content[0].text


async def eval_google_live(prompt: str, model: str) -> str:
    """Live Google API call."""
    from google import genai
    async with genai.Client().aio as client:
        response = await client.models.generate_content(model=model, contents=prompt)
    return response.text


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
                "temperature": 0
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

    async def _run():
        tasks = []
        task_info = []
        for case in cases:
            prompt = task_type.format_prompt(case)
            for provider in providers:
                if provider not in LIVE_EVAL_FNS:
                    continue
                model = models.get(provider)
                if not model:
                    continue
                fn = LIVE_EVAL_FNS[provider]
                tasks.append(fn(prompt, model))
                task_info.append((case, provider, model))
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for (case, provider, model), resp in zip(task_info, responses):
            if isinstance(resp, Exception):
                response = f"ERROR: {resp}"
                if task_type == CompositeTask:
                    answer = {"items": ["unknown"] * case["group_size"], "total_marked": -1}
                else:
                    answer = "unknown"
            else:
                response = resp
                answer = task_type.extract(response, case)
            results.append(task_type.make_result(case, provider, model, response, answer))
        return results

    results = asyncio.run(_run())

    if task_type == SingleTask:
        correct = sum(1 for r in results if r["correct"])
        print(f"\nResults: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
        for r in results:
            status = "✓" if r["correct"] else "✗"
            inp = r['input'][:25]
            print(f"  {status} {r['id']} ({r['provider']}): {inp}... → {r['extracted_answer']}")
    else:
        # Composite metrics
        n = len(results)
        all_correct = sum(1 for r in results if r["all_correct"])
        count_correct = sum(1 for r in results if r["count_correct"])
        avg_item_acc = sum(r["per_item_accuracy"] for r in results) / n if n > 0 else 0
        print(f"\nResults ({n} cases):")
        print(f"  Per-item accuracy: {100*avg_item_acc:.1f}%")
        print(f"  All-correct@8:     {all_correct}/{n} ({100*all_correct/n:.1f}%)")
        print(f"  Count exact match: {count_correct}/{n} ({100*count_correct/n:.1f}%)")
        for r in results:
            status = "✓" if r["all_correct"] else "✗"
            items = r["n_items_correct"]
            total = len(r["targets"])
            cnt = r["extracted_count"]
            tgt = r["target_count"]
            prov = r['provider']
            print(f"  {status} {r['id']} ({prov}): {items}/{total} items, count={cnt} (tgt {tgt})")
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

    is_composite = "all_correct" in results[0]

    if not is_composite:
        # SingleTask analysis
        by_model = defaultdict(lambda: {"correct": 0, "total": 0})
        by_diff = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

        for r in results:
            model = r["model"]
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
        model = r["model"]
        diff = r["difficulty"]

        by_model[model]["total"] += 1
        by_model[model]["item_acc_sum"] += r["per_item_accuracy"]
        if r["all_correct"]:
            by_model[model]["all_correct"] += 1
        if r["count_correct"]:
            by_model[model]["count_correct"] += 1

        by_diff[model][diff]["total"] += 1
        by_diff[model][diff]["item_acc_sum"] += r["per_item_accuracy"]
        if r["all_correct"]:
            by_diff[model][diff]["all_correct"] += 1
        if r["count_correct"]:
            by_diff[model][diff]["count_correct"] += 1

    def metrics(d):
        t = d["total"]
        if t == 0:
            return {"per_item_accuracy": 0, "all_correct_rate": 0, "count_match_rate": 0}
        return {
            "per_item_accuracy": d["item_acc_sum"] / t,
            "all_correct_rate": d["all_correct"] / t,
            "count_match_rate": d["count_correct"] / t,
            "total": t,
        }

    return {
        "by_model": {m: metrics(d) for m, d in by_model.items()},
        "by_difficulty": {
            m: {diff: metrics(d) for diff, d in diffs.items()}
            for m, diffs in by_diff.items()
        },
    }
