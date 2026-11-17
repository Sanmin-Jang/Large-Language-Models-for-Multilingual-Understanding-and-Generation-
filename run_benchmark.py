import os
import re
import json
import random
import time
import asyncio
import aiohttp
import pandas as pd

from dataclasses import dataclass, asdict
from collections import Counter
from typing import Optional
from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

OUTPUT_DIR = "benchmark_results"

LANGUAGE_QUOTAS = {
    "EN": 100,
    "RU": 100,
    "KZ": 100,
}

SOURCE_QUOTAS = {
    "EN": {
        "belebele": 50,
        "xnli": 50,
    },
    "RU": {
        "belebele": 40,
        "xnli": 30,
        "sberquad": 30,
    },
    "KZ": {
        "belebele": 100,
    },
}

MIN_TEXT_LENGTH = 20
MAX_TEXT_LENGTH = 5000

REQUEST_TIMEOUT = 90

OLLAMA_CONCURRENCY = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class BenchmarkItem:
    id: str
    language: str
    task: str
    source: str
    prompt: str
    expected_answer: str
    difficulty: str
    text_length: int


@dataclass
class EvaluationRecord:
    item_id: str
    language: str
    task: str
    source: str
    model: str
    expected: str
    predicted: str
    correct: bool
    score: float
    latency: float
    error: Optional[str] = None


# ============================================================
# TEXT PROCESSING & METRICS
# ============================================================

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def token_f1(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    common_count = sum(common.values())

    if common_count == 0:
        return 0.0

    precision = common_count / len(pred_tokens)
    recall = common_count / len(ref_tokens)

    return (2 * precision * recall) / (precision + recall)


def classify_difficulty(text_length):
    if text_length < 300:
        return "easy"
    elif text_length < 1000:
        return "medium"
    return "hard"


def valid_text(text):
    if not isinstance(text, str):
        return False
    length = len(text.strip())
    return MIN_TEXT_LENGTH <= length <= MAX_TEXT_LENGTH


def create_item(item_id, language, task, source, prompt, expected):
    combined_text = prompt + " " + str(expected)
    text_length = len(combined_text)

    return BenchmarkItem(
        id=str(item_id),
        language=language,
        task=task,
        source=source,
        prompt=prompt,
        expected_answer=str(expected),
        difficulty=classify_difficulty(text_length),
        text_length=text_length,
    )


# ============================================================
# DATASET LOADERS
# ============================================================

def load_belebele(language, amount):
    configs = {
        "EN": "eng_Latn",
        "RU": "rus_Cyrl",
        "KZ": "kaz_Cyrl",
    }
    config = configs[language]
    dataset = load_dataset("facebook/belebele", config, split="test")

    rows = list(dataset)
    random.shuffle(rows)
    items = []

    for row in rows:
        passage = row.get("flores_passage", "")
        question = row.get("question", "")
        answers = [
            row.get("mc_answer1", ""),
            row.get("mc_answer2", ""),
            row.get("mc_answer3", ""),
            row.get("mc_answer4", ""),
        ]
        correct_num = str(row.get("correct_answer_num", "")).strip()

        if not valid_text(passage) or not valid_text(question):
            continue
        if not all(isinstance(x, str) and x.strip() for x in answers):
            continue
        if correct_num not in {"1", "2", "3", "4"}:
            continue

        prompt = (
            "Read the passage and answer the question.\n\n"
            f"Passage:\n{passage}\n\n"
            f"Question:\n{question}\n\n"
            "Options:\n"
            f"A. {answers[0]}\n"
            f"B. {answers[1]}\n"
            f"C. {answers[2]}\n"
            f"D. {answers[3]}\n\n"
            "Return only the letter of the correct answer."
        )

        expected = ["A", "B", "C", "D"][int(correct_num) - 1]
        items.append(
            create_item(
                item_id=f"belebele_{language}_{len(items)}",
                language=language,
                task="QA_MULTIPLE_CHOICE",
                source="Belebele",
                prompt=prompt,
                expected=expected,
            )
        )
        if len(items) >= amount:
            break

    if len(items) < amount:
        raise RuntimeError(f"Belebele {language}: requested {amount}, got {len(items)}")
    return items


def load_xnli(language, amount):
    dataset = load_dataset("facebook/xnli", language.lower(), split="test")
    rows = list(dataset)
    random.shuffle(rows)

    labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
    items = []

    for row in rows:
        premise = row.get("premise", "")
        hypothesis = row.get("hypothesis", "")
        label = row.get("label")

        expected = label.lower() if isinstance(label, str) else labels.get(int(label), "")

        if not valid_text(premise) or not valid_text(hypothesis):
            continue
        if expected not in labels.values():
            continue

        prompt = (
            "Determine the relationship between the premise and the hypothesis.\n\n"
            f"Premise: {premise}\n\n"
            f"Hypothesis: {hypothesis}\n\n"
            "Choose exactly one label: entailment, neutral, contradiction.\n"
            "Return only the label."
        )

        items.append(
            create_item(
                item_id=f"xnli_{language}_{len(items)}",
                language=language,
                task="NLI",
                source="XNLI",
                prompt=prompt,
                expected=expected,
            )
        )
        if len(items) >= amount:
            break

    if len(items) < amount:
        raise RuntimeError(f"XNLI {language}: requested {amount}, got {len(items)}")
    return items


def load_sberquad(amount):
    dataset = load_dataset("kuznetsoffandrey/sberquad", "sberquad", split="validation")
    rows = list(dataset)
    random.shuffle(rows)
    items = []

    for row in rows:
        context = row.get("context", "")
        question = row.get("question", "")
        answers = row.get("answers", {})
        answer_list = answers.get("text", [])

        if not answer_list:
            continue
        expected = answer_list[0]

        if not valid_text(context) or not valid_text(question):
            continue
        if not isinstance(expected, str) or not expected.strip():
            continue

        prompt = (
            "Answer the question using the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Give a short answer based only on the context."
        )

        items.append(
            create_item(
                item_id=f"sberquad_RU_{len(items)}",
                language="RU",
                task="QA_EXTRACTIVE",
                source="SberQuAD",
                prompt=prompt,
                expected=expected,
            )
        )
        if len(items) >= amount:
            break

    if len(items) < amount:
        raise RuntimeError(f"SberQuAD: requested {amount}, got {len(items)}")
    return items


def build_benchmark():
    all_items = []
    for language, sources in SOURCE_QUOTAS.items():
        for source, amount in sources.items():
            if source == "belebele":
                all_items.extend(load_belebele(language, amount))
            elif source == "xnli":
                all_items.extend(load_xnli(language, amount))
            elif source == "sberquad":
                all_items.extend(load_sberquad(amount))

    unique_items = []
    seen = set()
    for item in all_items:
        key = (item.language, normalize_text(item.prompt))
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(item)

    counts = Counter(item.language for item in unique_items)
    for language, expected in LANGUAGE_QUOTAS.items():
        if counts[language] != expected:
            raise RuntimeError(f"Expected {expected} items for {language}, got {counts[language]}")

    random.shuffle(unique_items)
    dataset_path = os.path.join(OUTPUT_DIR, "benchmark_dataset.jsonl")
    with open(dataset_path, "w", encoding="utf-8") as file:
        for item in unique_items:
            file.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    return unique_items


# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_answer(text, task):
    text = text.strip()
    if task == "QA_MULTIPLE_CHOICE":
        match = re.search(r"\b([ABCD])\b", text.upper())
        return match.group(1) if match else text

    if task == "NLI":
        text_lower = text.lower()
        for label in ["entailment", "contradiction", "neutral"]:
            if label in text_lower:
                return label
        return text_lower

    return text


# ============================================================
# OLLAMA API
# ============================================================

async def call_ollama(session, item, semaphore):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are an evaluation assistant. Follow the output format strictly. Be brief."},
            {"role": "user", "content": item.prompt},
        ],
        "options": {
            "temperature": 0,
            "num_predict": 25,
        },
        "stream": False,
    }

    async with semaphore:
        async with session.post(OLLAMA_URL, json=payload) as response:
            data = await response.json()
            if response.status != 200:
                raise RuntimeError(f"Ollama error {response.status}: {data}")
            return data["choices"][0]["message"]["content"]


# ============================================================
# EVALUATION WORKER
# ============================================================

def evaluate_prediction(item, prediction):
    predicted = extract_answer(prediction, item.task)
    expected = item.expected_answer

    if item.task in {"QA_MULTIPLE_CHOICE", "NLI"}:
        correct = normalize_text(predicted) == normalize_text(expected)
        return predicted, correct, float(correct)

    score = token_f1(predicted, expected)
    correct = score >= 0.5
    return predicted, correct, score


async def evaluate_item(session, item, semaphore):
    start_time = time.perf_counter()
    try:
        raw_response = await call_ollama(session, item, semaphore)
        latency = time.perf_counter() - start_time
        predicted, correct, score = evaluate_prediction(item, raw_response)
        return EvaluationRecord(
            item_id=item.id,
            language=item.language,
            task=item.task,
            source=item.source,
            model=OLLAMA_MODEL,
            expected=item.expected_answer,
            predicted=predicted,
            correct=correct,
            score=score,
            latency=latency,
        )
    except Exception as error:
        latency = time.perf_counter() - start_time
        return EvaluationRecord(
            item_id=item.id,
            language=item.language,
            task=item.task,
            source=item.source,
            model=OLLAMA_MODEL,
            expected=item.expected_answer,
            predicted="",
            correct=False,
            score=0.0,
            latency=latency,
            error=str(error),
        )


# ============================================================
# STATS & REPORT
# ============================================================

def bootstrap_confidence_interval(values, iterations=2000):
    if not values:
        return 0.0, 0.0
    rng = random.Random(SEED)
    means = [sum(rng.choices(values, k=len(values))) / len(values) for _ in range(iterations)]
    means.sort()
    return means[int(0.025 * iterations)], means[int(0.975 * iterations)]


def calculate_metrics(records):
    valid = [r for r in records if r.error is None]
    if not valid:
        return {"total": len(records), "successful": 0, "accuracy": 0.0, "average_score": 0.0, "average_latency": 0.0}
    return {
        "total": len(records),
        "successful": len(valid),
        "accuracy": round(sum(r.correct for r in valid) / len(valid), 4),
        "average_score": round(sum(r.score for r in valid) / len(valid), 4),
        "average_latency": round(sum(r.latency for r in valid) / len(valid), 3),
    }


def generate_report(records):
    df = pd.DataFrame([asdict(r) for r in records])
    csv_path = os.path.join(OUTPUT_DIR, "detailed_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    valid_records = [r for r in records if r.error is None]
    acc_values = [1.0 if r.correct else 0.0 for r in valid_records]
    ci_lower, ci_upper = bootstrap_confidence_interval(acc_values)

    report = {
        "model": OLLAMA_MODEL,
        "overall": calculate_metrics(records),
        "accuracy_95_ci": [round(ci_lower, 4), round(ci_upper, 4)],
        "by_language": {lang: calculate_metrics([r for r in records if r.language == lang]) for lang in ["EN", "RU", "KZ"]},
        "by_task": {t: calculate_metrics([r for r in records if r.task == t]) for t in sorted(df["task"].unique())},
        "by_source": {s: calculate_metrics([r for r in records if r.source == s]) for s in sorted(df["source"].unique())},
    }

    json_path = os.path.join(OUTPUT_DIR, "summary_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n========== FINAL REPORT ==========")
    print(json.dumps(report, indent=4, ensure_ascii=False))
    print(f"\nFiles saved:\n- {csv_path}\n- {json_path}")


# ============================================================
# MAIN
# ============================================================

async def main():
    benchmark = build_benchmark()
    total_items = len(benchmark)

    print("\nStarting Ollama evaluation...")
    print("Ollama model:", OLLAMA_MODEL)
    print("Benchmark size:", total_items)

    semaphore = asyncio.Semaphore(OLLAMA_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    records = []
    completed = 0

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def runner(item):
            nonlocal completed
            rec = await evaluate_item(session, item, semaphore)
            completed += 1
            if completed % 10 == 0 or completed == total_items:
                print(f"Progress: [{completed}/{total_items}] ({rec.language} | {rec.task})")
            if rec.error:
                print(f"  [Error] {item.id}: {rec.error}")
            return rec

        tasks = [runner(item) for item in benchmark]
        records = await asyncio.gather(*tasks)

    generate_report(records)
    print("\nBenchmark completed.")


if __name__ == "__main__":
    asyncio.run(main())