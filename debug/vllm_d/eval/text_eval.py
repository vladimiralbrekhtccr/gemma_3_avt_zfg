import json
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import requests

# ----- Your existing config -----
SERVED_NAME = "kita"
BASE_URL = "http://localhost:6655/v1"

all_langs = {
    "kk": "Kazakh",
    "en": "English",
    "ru": "Russian",
}

def _api_root_from_base(base_url: str) -> str:
    return base_url.split("/v1")[0] if "/v1" in base_url else base_url

_API_ROOT = _api_root_from_base(BASE_URL)
_TOKENIZE_ENDPOINT = f"{_API_ROOT}/tokenize"

def _token_count(text: str) -> int:
    try:
        r = requests.post(_TOKENIZE_ENDPOINT, json={"prompt": text}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "count" in data and isinstance(data["count"], int):
            return data["count"]
        if "tokens" in data and isinstance(data["tokens"], list):
            return len(data["tokens"])
    except Exception:
        pass
    return 0

def _make_user_prompt(input_text: str, tgt_lang: str, src_lang=None, tgt_mode='text') -> str:
    instruction = f'Translate the following text into {all_langs[tgt_lang]}.'
    NEWLINE = '\n'
    if NEWLINE in input_text:
        instruction += f' Preserve every {NEWLINE} token—same count.'
    if tgt_mode == 'speech':
        instruction = instruction + " Transcribe all numbers as read."
    if tgt_mode == 'speech' and src_lang and src_lang == tgt_lang:
        instruction = f"Do not translate or change this {all_langs[src_lang]} text, only transcribe all numbers as read."
    return f"{instruction}\n\n{input_text}"

def get_prediction(input_text, tgt_lang, src_lang=None, tgt_mode='text'):
    client = OpenAI(api_key="empty", base_url=BASE_URL)  # local client per call (thread-safe)
    _input = _make_user_prompt(input_text, tgt_lang, src_lang, tgt_mode)
    resp = client.chat.completions.create(
        model=SERVED_NAME,
        messages=[{"role": "user", "content": _input}],
        temperature=0.05,
        top_p=0.95,
        max_tokens=100,
        frequency_penalty=0.3,
    )
    return resp.choices[0].message.content

# ---------- NEW: оценка скорости по списку ----------
def _percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

def evaluate_list_to_jsonl(
    texts_list,
    out_jsonl_path,
    tgt_lang="kk",
    max_workers=100,
    max_retries=3,
    retry_backoff=0.7,
):
    """
    Берёт СПИСОК строк `texts_list`, отправляет до `max_workers` параллельных запросов,
    и пишет результаты (и время обработки запроса) в JSONL:
        {"index": ..., "original_text": ..., "gemma_translation": ..., "latency_ms": ..., "ok": ...}
    Возвращает метрики со стендовым временем (wall time).
    """

    def worker(idx, src_text):
        last_err = None
        for attempt in range(1, max_retries + 1):
            start = time.perf_counter()
            try:
                tr = get_prediction(src_text, tgt_lang)
                latency_ms = (time.perf_counter() - start) * 1000.0
                return {"index": idx, "original_text": src_text,
                        "gemma_translation": tr, "latency_ms": latency_ms, "ok": True}
            except Exception as e:
                last_err = e
                time.sleep(retry_backoff * attempt)
        latency_ms = (time.perf_counter() - start) * 1000.0  # время последней попытки
        return {"index": idx, "original_text": src_text, "error": str(last_err),
                "latency_ms": latency_ms, "ok": False}

    futures = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex, \
         open(out_jsonl_path, "w", encoding="utf-8") as fout:

        for idx, src_text in enumerate(texts_list):
            futures.append(ex.submit(worker, idx, src_text))

        latencies = []
        ok_cnt = 0

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            rec = fut.result()
            if rec.get("ok"):
                ok_cnt += 1
                if rec.get("latency_ms") is not None:
                    latencies.append(rec["latency_ms"])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_time_s = time.perf_counter() - t0
    eff_rps = (ok_cnt / total_time_s) if total_time_s > 0 else float("inf")
    latencies.sort()

    metrics = {
        "total_requests": len(texts_list),
        "successful": ok_cnt,
        "failed": len(texts_list) - ok_cnt,
        "total_time_s": total_time_s,
        "effective_rps": eff_rps,
        "latency_ms": {
            "avg": (sum(latencies) / len(latencies)) if latencies else float("nan"),
            "min": latencies[0] if latencies else float("nan"),
            "max": latencies[-1] if latencies else float("nan"),
            "p50": _percentile(latencies, 50) if latencies else float("nan"),
            "p90": _percentile(latencies, 90) if latencies else float("nan"),
            "p99": _percentile(latencies, 99) if latencies else float("nan"),
            "count": len(latencies),
        },
        "max_workers": max_workers,
    }

    print(f"\nDone. Wrote JSONL to: {out_jsonl_path}")
    print(f"Total Time: {metrics['total_time_s']:.2f} seconds")
    print(f"Effective RPS: {metrics['effective_rps']:.2f} req/s\n")
    print("Request Latency (ms):")
    lm = metrics["latency_ms"]
    print(f"  Avg: {lm['avg']:.1f}, Min: {lm['min']:.1f}, Max: {lm['max']:.1f}")
    print(f"  P50: {lm['p50']:.1f}, P90: {lm['p90']:.1f}, P99: {lm['p99']:.1f}")

    return metrics

# --------- NEW: после генерации — считаем токены через /tokenize ---------
def summarize_tokens_with_tokenizer_api(
    jsonl_path: str,
    tgt_lang: str,
    total_time_s: float,
    src_lang=None,
    tgt_mode='text',
):
    total_input_tokens = 0
    total_output_tokens = 0
    ok = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            ok += 1
            original = rec.get("original_text") or ""
            output = rec.get("gemma_translation") or ""
            # ВОССТАНАВЛИВАЕМ ровно тот user-подсказ, что отправляли:
            prompt_text = _make_user_prompt(original, tgt_lang, src_lang, tgt_mode)
            total_input_tokens += _token_count(prompt_text)
            total_output_tokens += _token_count(output)

    input_tps = (total_input_tokens / total_time_s) if total_time_s > 0 else float("inf")
    output_tps = (total_output_tokens / total_time_s) if total_time_s > 0 else float("inf")
    total_tps = ((total_input_tokens + total_output_tokens) / total_time_s) if total_time_s > 0 else float("inf")

    print("\nTokens (via /tokenize):")
    print(f"  Successful requests     : {ok}")
    print(f"  Total Input Tokens      : {total_input_tokens}")
    print(f"  Total Generated Tokens  : {total_output_tokens}")
    print(f"  Input (tokens/s)        : {input_tps:.2f}")
    print(f"  Output (tokens/s)       : {output_tps:.2f}")
    print(f"  Total Throughput        : {total_tps:.2f} tokens/s")

    return {
        "successful": ok,
        "total_input_tokens": total_input_tokens,
        "total_generated_tokens": total_output_tokens,
        "input_tokens_per_sec": input_tps,
        "output_tokens_per_sec": output_tps,
        "total_throughput_tokens_per_sec": total_tps,
    }

# --------- Example run with LIST ---------
if __name__ == "__main__":
    QUESTIONS = [
        "Какое у тебя любимое аниме? /no_think",
        "Что важнее в аниме: сюжет или визуал? /no_think",
        "Какое аниме ты бы порекомендовал новичку? /no_think",
        "Какая опенинг-песня у тебя в топе? /no_think",
        "Какой персонаж тебя вдохновляет? /no_think",
    ]

    N = 100
    TEXTS = (QUESTIONS * math.ceil(N / len(QUESTIONS)))[:N]
    OUT = "translations_from_list.jsonl"

    metrics = evaluate_list_to_jsonl(TEXTS, OUT, tgt_lang="kk", max_workers=128)

    # После генерации считаем токены через /tokenize и печатаем сводку
    summarize_tokens_with_tokenizer_api(
        jsonl_path=OUT,
        tgt_lang="kk",
        total_time_s=metrics["total_time_s"],
    )
