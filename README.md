# agentic-memory-long-context-bench

A synthetic benchmark dataset and generator for testing whether memory systems help LLM agents when relevant information is buried inside long, noisy histories.

This repo is designed to answer a specific question:

> when task-relevant context is larger than what is practical to include in every prompt, does retrieval-based memory outperform prompt stuffing?

## What This Repo Contains

- a deterministic dataset generator
- a JSONL dataset format for long multi-turn conversations
- scenario types that stress profile recall, troubleshooting continuity, contradiction handling, and procedure reuse
- a benchmark-ready schema that can be consumed by a separate evaluator
- explicit token-budget tiers so examples can exceed practical or absolute context limits

This repo is only about dataset generation and dataset definition.
The evaluation runner can live here later, or in a separate repo that compares:

- short-context baseline
- full-context baseline
- memory-enabled system

The repo now also includes a first harness implementation for those three modes.

## Live Benchmark Artifacts

The first full benchmark run against the `>300k` token dataset is now checked into the repo.

Core assets:

- Dataset: [`datasets/long_context_v2_300k.jsonl`](datasets/long_context_v2_300k.jsonl)
- Raw results: [`results/beyond_300k_full_run/results.jsonl`](results/beyond_300k_full_run/results.jsonl)
- Resume checkpoint: [`results/beyond_300k_full_run/checkpoint.jsonl`](results/beyond_300k_full_run/checkpoint.jsonl)
- Markdown report: [`reports/beyond_300k_full_run/benchmark_report.md`](reports/beyond_300k_full_run/benchmark_report.md)
- PDF report: [`reports/beyond_300k_full_run/benchmark_report.pdf`](reports/beyond_300k_full_run/benchmark_report.pdf)
- Summary JSON: [`reports/beyond_300k_full_run/benchmark_summary.json`](reports/beyond_300k_full_run/benchmark_summary.json)
- Example breakdown CSV: [`reports/beyond_300k_full_run/example_breakdown.csv`](reports/beyond_300k_full_run/example_breakdown.csv)

Headline results from that run:

- `memory_enabled`: `7/12` passed (`58.3%`)
- `full_context`: `3/12` passed (`25.0%`)
- `short_context`: `0/12` passed (`0.0%`)
- `memory_enabled` used `289.2` average prompt tokens vs `162590.2` for `full_context`
- `memory_enabled` cost about `152.2x` less per example than `full_context`

Important caveat:

- the current `12`-example run is a pilot, not the final public benchmark
- the report now includes confidence intervals and should be interpreted as directional evidence
- the next benchmark target should be at least `100` examples total, with `20+` examples per scenario family
- for tighter 95% pass-rate intervals around `+/-5` percentage points, target roughly `385-400` examples total

## Dataset Design

Each example is a long conversation with:

- durable user facts
- ephemeral events
- distractor turns
- superseded facts
- optional reusable procedures
- a final task that depends on non-local context

Each example also includes structured ground truth:

- gold facts that should be used
- stale facts that should be ignored
- expected answer constraints
- the expected evidence set

## Scenario Families

- `profile_recall`
  The answer requires recalling user profile facts mentioned far earlier.

- `troubleshooting_continuity`
  The answer should not repeat steps the user already tried.

- `contradiction_resolution`
  Earlier facts are superseded by later corrections; stale facts should be ignored.

- `procedure_reuse`
  The answer should apply a previously successful support or ops procedure.

- `mixed_long_context`
  Durable facts, episodic events, stale facts, and procedures are all present together.

## Why Synthetic?

We want:

- reproducibility
- exact control over where facts appear
- exact control over distractor density
- exact gold labels for scoring

Real chat logs are useful later, but synthetic fixtures make it much easier to measure quality, hallucination, latency, and cost cleanly.

## JSONL Schema

Each line in `datasets/*.jsonl` is a single benchmark example.

Top-level fields:

- `id`
- `seed`
- `scenario_type`
- `difficulty`
- `conversation`
- `task`
- `gold`
- `metadata`

Important `metadata` fields include:

- `estimated_transcript_tokens`
- `supporting_fact_tokens`
- `distractor_tokens`
- `context_tier`
- `target_min_tokens`

`conversation` is a list of turns:

```json
{
  "role": "user",
  "turn_index": 17,
  "kind": "durable_fact",
  "text": "My timezone is UTC+5:30."
}
```

`task` describes the final question:

```json
{
  "prompt": "Summarize the user's current issue and propose the next best action.",
  "requires": [
    "current_issue",
    "attempted_steps",
    "current_plan"
  ]
}
```

`gold` stores exact evaluation targets:

```json
{
  "must_include": ["enterprise", "webhook signature", "replay"],
  "must_not_include": ["starter", "clear browser cache"],
  "supporting_fact_ids": ["fact_3", "fact_9", "proc_1"],
  "stale_fact_ids": ["fact_2"]
}
```

## Quickstart

Generate a small sample dataset:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
generate-long-context-dataset --examples 25 --output datasets/sample_v1.jsonl
```

Generate a larger dataset:

```bash
generate-long-context-dataset --examples 250 --seed 7 --output datasets/long_context_v1.jsonl
```

Generate a dataset that explicitly targets transcripts beyond a 250k-token budget:

```bash
generate-long-context-dataset \
  --examples 12 \
  --seed 7 \
  --min-tokens 300000 \
  --context-tier beyond_250k \
  --output datasets/long_context_v2_300k.jsonl
```

Run the harness against a dataset:

```bash
export GEMINI_API_KEY=your_key_here
run-long-context-harness \
  --dataset datasets/long_context_v2_300k.jsonl \
  --output results \
  --run-name benchmark_v1 \
  --model gemini-2.5-flash \
  --judge-model gemini-2.5-flash-lite \
  --sleep-between-requests 1.5 \
  --resume \
  --limit 3 \
  --report
```

Harness modes:

- `short_context`
- `full_context`
- `memory_enabled`

The `memory_enabled` mode explicitly uses:

- semantic memory for durable facts and corrections
- episodic memory for events and attempted steps
- procedural memory for prior successful procedures

and records the retrieved memory traces in the output JSONL.

The transcript-based modes and memory-based mode now share the same answer scaffold:

- identify relevant facts first
- prefer newer corrected facts over stale ones
- avoid repeating already-tried troubleshooting steps
- return `RELEVANT_FACTS` followed by `ANSWER`

That change makes the `full_context` baseline meaningfully fairer than raw transcript stuffing for the next benchmark rerun.

Each run gets its own folder, for example:

```text
results/
└── benchmark_v1/
    ├── checkpoint.jsonl
    └── results.jsonl
```

That makes the output live somewhere clean and resumable.

Generate a report bundle from a completed run:

```bash
generate-benchmark-report \
  --results results/benchmark_v1/results.jsonl \
  --output-dir reports/benchmark_v1
```

That creates:

```text
reports/
└── benchmark_v1/
    ├── benchmark_report.md
    ├── benchmark_report.pdf
    ├── benchmark_summary.json
    └── example_breakdown.csv
```

## Generation Strategy

The generator does not ask an LLM to write the dataset.

Instead it uses seeded templates and deterministic slot-filling:

- choose a scenario family
- sample entities and facts from controlled vocabularies
- place important facts early or mid-conversation
- inject distractor turns later
- optionally expand the transcript with large distractor blocks until a target token budget is reached
- optionally supersede one fact with a correction
- insert prior procedure outcomes
- create a final task whose gold answer depends on the planted evidence

That gives us exact labels and stable regeneration.

## Next Step
The current harness writes per-example results with:

- rule-based pass/fail scoring
- optional LLM-as-judge scoring
- latency
- prompt/completion tokens
- estimated cost
- memory traces for semantic, episodic, and procedural retrieval

Operational safeguards:

- retries with exponential backoff on transient failures and rate limits
- optional sleep between requests
- checkpointing so interrupted runs can resume
