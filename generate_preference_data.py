"""
Part 1 -- build your preference data.

You construct at least MIN_PREFERENCE_PAIRS (1,000) preference pairs over two input
classes:

  - ambiguous: a request that is genuinely underspecified. The PREFERRED (chosen)
    response asks a specific clarifying question; the DISPREFERRED (rejected) response
    is a confident guess that commits to one interpretation.

  - clear: a fully-specified request. The PREFERRED (chosen) response answers directly;
    the DISPREFERRED (rejected) response is a needless clarifying question.

You MAY synthesize these with an LLM. The course model (Llama-3.1-8B-Instruct) runs on
PACE-ICE at no cost -- load it locally the same way eval/judge.py's LocalHFJudge does.
You MUST hand-verify a sample of HAND_VERIFY_SAMPLE (100) pairs and report the
rejection rate of your own synthetic data (spec, Part 1 + the "preference data quality" rubric line).

What is wired for you: the output schema + writer, the verification sampler, and the
rejection-rate computation. What you implement: `synthesize_pairs()` -- the actual data generation.

INTEGRITY: build your own prompts. Do NOT reuse the held-out eval prompts in
eval/held_out_*.jsonl as training data -- that is training on the test set.

Usage:
    python3 generate_preference_data.py --generate   # Synthesize + write pairs, then
                                                      # sample 100 for hand-verification.

    # ... now open data/hand_verified.jsonl and fill in each "verdict" ...
    python3 generate_preference_data.py --report      # Compute + print the rejection rate.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
# from xml.parsers.expat import model 

from configs.train_config import (
    DATA_SEED,
    HAND_VERIFY_SAMPLE,
    MIN_PREFERENCE_PAIRS,
)
from configs.paths_config import HAND_VERIFIED_PATH, PREFERENCE_DATA_PATH

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

VALID_CLASSES = {"ambiguous", "clear"}


def synthesize_pairs(n: int) -> list[dict]:
    """
    TODO (student): produce `n` preference pairs, balanced across the two classes.

    Return a list of dicts, each:
        {"class": "ambiguous" | "clear",
         "prompt": "<user request>",
         "chosen": "<preferred response>",
         "rejected": "<dispreferred response>"}

    (The "id" field is added for you by write_pairs().)

    Approach is up to you. A common recipe: prompt an LLM to generate ambiguous and
    clear requests across varied domains, then for each request generate both a
    clarifying-question response and a direct-answer response, and assign chosen/rejected
    by the request's class. Keep DATA_SEED fixed wherever you sample, so your data is reproducible.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1")
    
    batch_size = 10
    ambiguous_target = (n + 1) // 2
    clear_target = n // 2 
    
    # load model
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    # print(f"Loading tokenizer: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # print("Tokenizer loaded.", flush=True)
    
    # print("Loading model; this may take a while...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        )
    model.eval()
    # print("Model loaded.", flush=True)
    
    #start generating data
    pairs = []
    
    for class_name, target in (
        ("ambiguous", ambiguous_target),
        ("clear", clear_target)
    ):
        class_pairs = []
        max_consecutive_failures = 5
        consecutive_failures = 0
        while len(class_pairs) < target:
            remaining = target - len(class_pairs)
            request_count = min(batch_size, remaining)
            # let LLM create data with request_count requests for class_name
            if class_name == "ambiguous":
                response_fields = ("clarify", "guess")
                task = (f"""
                    Generate genuinely ambiguous user requests.

                    A request is ambiguous only when an important missing detail would materially
                    change a useful, tailored response. Do not label a request ambiguous just because
                    the assistant could ask an optional follow-up question.

                    For each object:
                    - "prompt": a realistic user request with important information omitted.
                    - "clarify": one concise, specific question asking for the missing information
                    needed to tailor the response.
                    - "guess": a plausible, useful answer that clearly commits to one reasonable
                    unstated assumption. Do not invent facts or present a guess as verified.

                    The prompt must make the missing information apparent. Prefer varied everyday
                    tasks such as planning, writing, recommendations, troubleshooting, and coding.
                    Do not use simple factual questions that have a direct, stable answer.
                """)
            else: # clear
                response_fields = ("direct", "needless")
                task = ("""
                    Generate fully specified user requests that can be
                    answered directly and correctly using information in the prompt and stable,
                    widely known facts.

                    For each object:
                    - "prompt": a natural, sufficiently specified user request.
                    - "direct": a concise, correct answer that actually fulfills the request;
                    do not merely repeat the prompt.
                    - "needless": one plausible but unnecessary clarification question. The prompt
                    already contains enough information to answer, so this question should delay
                    rather than help complete the task.

                    Use varied task types and domains, not only list questions. Avoid jokes,
                    riddles, trick questions, subjective "best" questions, current facts, changing
                    statistics, rankings, and questions whose answer depends on a source or date.
                """)
            messages = [
                {
                    "role": "system", 
                    "content": f"""
                        You create high-quality preference-training examples for a conversational assistant.
                        Return only a valid JSON array. Do not include Markdown, code fences, comments,
                        or text outside the JSON.

                        Each object must contain exactly the requested keys, and every value must be
                        a non-empty string. Generate exactly the requested number of objects.

                        Use natural, varied requests across different domains and task types. Avoid
                        duplicate or near-duplicate requests. Use stable, widely known facts only;
                        avoid current events, current weather, prices, populations, rankings, and
                        claims that depend on an unspecified source or date.
                        """
                },
                {
                    "role": "user",
                    "content": (
                        f"Generate exactly {request_count} records. {task}\n"
                        f"Each record must have exactly these keys: "
                        f"'prompt', '{response_fields[0]}', '{response_fields[1]}'."
                    )
                },
            ]
            
            # tokenize the messages and generate text
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            # print(
            #     f"[{class_name}] Generating {request_count} record(s)...",
            #     flush=True,
            # )
            
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    )
                
            # print(f"[{class_name}] Generation finished.", flush=True)    
                
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            # print(generated_text, flush=True)
            # print(f"[{class_name}] Decoded output.", flush=True)
            
            # check the response and turn into pairs with chosen/rejected
            valid_records = []
            required_fields = {"prompt", *response_fields}
            
            try:
                records = json.loads(generated_text)
            except json.JSONDecodeError as e:
                print(f"Generated text is not valid JSON, will retry. Error: {e}")
                records = None
            
            if not isinstance(records, list):
                print("Generated output is not a JSON array, will retry.")
            else:
                for record in records:
                    if not isinstance(record, dict):
                        # raise ValueError(f"Expected a JSON object, got: {record}")
                        continue
                    if not required_fields.issubset(record.keys()):
                        # raise ValueError(f"Record #{records.index(record)} missing keys {sorted(required_fields - record.keys())}")
                        continue
                    if any(
                        not isinstance(record[field], str) or not record[field].strip() 
                        for field in required_fields):
                        # raise ValueError(f"Record #{records.index(record)} has invalid values for keys {sorted(required_fields)}")
                        continue
                    valid_records.append(record)
                
            records_to_add = valid_records[:request_count]

            for record in records_to_add:
                if class_name == "ambiguous":
                    chosen, rejected = record["clarify"], record["guess"]
                else:
                    chosen, rejected = record["direct"], record["needless"]
                    
                class_pairs.append({
                    "class": class_name,
                    "prompt": record["prompt"],
                    "chosen": chosen,
                    "rejected": rejected,
                })
            
            # check the progress
            remaining = target - len(class_pairs)
            print(
                f"{class_name}: this time adds {len(records_to_add)} pairs,"
                f"progress {len(class_pairs)}/{target},still remain {remaining} pairs"
            )
            # calculate consecutive failures
            if not records_to_add:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    raise RuntimeError(
                        f"Failed to generate valid records for {class_name} in continuous {max_consecutive_failures} times." 
                        f"please check the prompt"
                    )
            else:
                consecutive_failures = 0    
            
        # append to class_pairs
        pairs.extend(class_pairs)
    
    # output list
    return pairs
    
    # raise NotImplementedError("Implement preference-pair synthesis.")


# --- Wired: everything below is provided -------------------------------------------


def write_pairs(pairs: list[dict], path: str | Path = PREFERENCE_DATA_PATH) -> None:
    """Validates, assigns stable ids, and writes pairs as JSONL."""
    required = {"class", "prompt", "chosen", "rejected"}

    for i, p in enumerate(pairs):
        missing = required - p.keys()

        if missing:
            raise ValueError(f"pair #{i} missing keys {sorted(missing)}")

        if p["class"] not in VALID_CLASSES:
            raise ValueError(f"pair #{i} has invalid class {p['class']!r}")

    if len(pairs) < MIN_PREFERENCE_PAIRS:
        raise ValueError(
            f"Only {len(pairs)} pairs; the assignment requires at least " f"{MIN_PREFERENCE_PAIRS}."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            rec = {
                "id": f"pair_{i:06d}",
                "class": p["class"],
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_amb = sum(1 for p in pairs if p["class"] == "ambiguous")
    n_clr = sum(1 for p in pairs if p["class"] == "clear")
    print(f"Wrote {len(pairs)} pairs ({n_amb} ambiguous, {n_clr} clear) -> {path}")


def sample_for_verification(
    pairs_path: str | Path = PREFERENCE_DATA_PATH,
    out_path: str | Path = HAND_VERIFIED_PATH,
    k: int = HAND_VERIFY_SAMPLE,
    seed: int = DATA_SEED,
) -> None:
    """Samples k pairs into a hand-verification file with a blank 'verdict' field.

    Open the file and set each pair's "verdict" to "accept" or "reject" by hand
    (a pair is a reject if the chosen/rejected labels are wrong, the prompt is
    mislabeled ambiguous/clear, or either response is low quality). Then run
    --report.
    """
    pairs = [
        json.loads(l)
        for l in Path(pairs_path).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    rng = random.Random(seed)
    sample = rng.sample(pairs, min(k, len(pairs)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in sample:
            rec = dict(rec)
            rec["verdict"] = ""  # Fill with "accept" or "reject" by hand.
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sample)} pairs to verify -> {out_path}")
    print("Fill in each 'verdict' (accept/reject) by hand, then run --report.")


def rejection_rate(path: str | Path = HAND_VERIFIED_PATH) -> float:
    """Computes the rejection rate over the hand-verified sample."""
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    verdicts = [str(r.get("verdict", "")).strip().lower() for r in rows]

    unfilled = [i for i, v in enumerate(verdicts) if v not in {"accept", "reject"}]
    if unfilled:
        raise ValueError(
            f"{len(unfilled)} of {len(rows)} verdicts are not 'accept'/'reject' "
            f"(first at row {unfilled[0]}). Finish hand-verifying before reporting."
        )

    n_reject = sum(1 for v in verdicts if v == "reject")
    rate = n_reject / len(rows)

    print(f"Hand-verified {len(rows)} pairs: {n_reject} rejected -> rejection rate {rate:.1%}")
    return rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="synthesize + write pairs, then sample for verification",
    )

    parser.add_argument(
        "--n", type=int, default=MIN_PREFERENCE_PAIRS, help="how many pairs to synthesize"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="compute the rejection rate from the hand-verified file",
    )

    args = parser.parse_args()

    if args.generate:
        pairs = synthesize_pairs(args.n)
        write_pairs(pairs)
        sample_for_verification()

    if args.report:
        rejection_rate()

    if not (args.generate or args.report):
        parser.print_help()


if __name__ == "__main__":
    main()
