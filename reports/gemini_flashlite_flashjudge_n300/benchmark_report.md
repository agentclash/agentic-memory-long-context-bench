# agentic-memory n=300 Long-Context Benchmark Report

On >300k-token conversations, agentic-memory retrieval outperformed both short-context and full-context baselines while using dramatically fewer prompt tokens than transcript stuffing.

## Benchmark Setup

- Dataset: `300` examples from the `beyond_250k` tier
- Transcript size: approximately `300040` to `318607` estimated tokens per example
- Model under test: `gemini-2.5-flash-lite`
- Judge model: `gemini-3-flash-preview`
- Total mode runs: `900`

## Results

| Mode | Pass Rate | Avg Judge | Avg Latency (ms) | Avg Prompt Tokens | Avg Cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_enabled` | 171/300 (57.0%) | 0.7338 | 1045.49 | 349.4 | $0.000066 |
| `full_context` | 72/300 (24.0%) | 0.3234 | 2511.77 | 160543.9 | $0.016114 |
| `short_context` | 4/300 (1.3%) | 0.1137 | 979.73 | 1718.4 | $0.000196 |

## Confidence Intervals

| Mode | 95% Wilson CI For Pass Rate |
| --- | ---: |
| `memory_enabled` | 51.3% to 62.5% |
| `full_context` | 19.5% to 29.1% |
| `short_context` | 0.5% to 3.4% |

## Good Enough?

Yes, with boundaries. The completed run is strong enough to support a narrow claim that retrieval-based memory beats transcript stuffing on this synthetic long-context benchmark. It is not yet strong enough to support the broadest production or competitor-comparison claims.

## What You Can Confidently Claim

1. Memory retrieval beats transcript stuffing on recall-heavy long-context tasks. Overall, `memory_enabled` reached `57.0%` versus `24.0%` for `full_context`, a gap of `33.0` percentage points.
2. Contradiction resolution is the single strongest result in the benchmark. `memory_enabled` solved `60/60` (`100.0%`) while `full_context` solved `15/60` (`25.0%`).
3. The economics are unambiguous. `memory_enabled` used `349.4` prompt tokens versus `160543.9` for `full_context`, and cost about `242.5x` less per row.
4. Profile recall works. `memory_enabled` solved `42/60` (`70.0%`) while `full_context` solved `0/60` (`0.0%`).
5. Short context is effectively useless for these tasks. It finished at `1.3%` overall.
6. The benchmark infrastructure itself works: deterministic dataset generation, resumable execution, and 900 completed rows.

## What You Cannot Claim

1. Do not claim that memory reduces hallucination in a broad sense. The current failure label is really stale/forbidden-fact citation under strict must-not-include rules, not a clean fabrication metric.
2. Do not claim memory beats full context across all task types. In troubleshooting continuity, `full_context` scored `95.0%` versus `83.3%` for `memory_enabled`.
3. Do not claim procedural memory is already a strong differentiator. On procedure reuse, `memory_enabled` scored only `30.0%`.
4. Do not claim this generalizes to production as-is. The memory path still benefits from benchmark turn-kind labels at ingestion time.
5. Do not claim this works across model families in general. The completed run used a single participant model family.

## What Is Genuinely Unanswered

1. How much perfect ingestion labeling inflates the result remains the biggest open question.
2. Why mixed long-context collapses so badly is unresolved: `memory_enabled` scored `1.7%` there.
3. Whether the semantic / episodic / procedural taxonomy itself matters is still unknown without ablations.
4. This benchmark does not yet compare against actual memory products such as Mem0, Supermemory, or Zep.
5. The vocabulary and distractors likely need expansion before making the broadest generalization claims.

## Claim Status

| Claim | Status |
| --- | --- |
| Memory retrieval beats stuffing on recall tasks | Say it |
| Contradiction resolution is a killer feature | Say it loudly |
| 244x cost reduction | Say it |
| Reduces hallucination | Don't say it |
| Works for all task types | Don't say it |
| Procedural memory is strong | Don't say it yet |
| Better than Mem0 | Can't say it |
| Works in production without labels | Can't say it |

## Why This Matters

- `memory_enabled` improved pass rate by `33.0` percentage points over `full_context`.
- `memory_enabled` reduced prompt-token volume by `99.8`% versus `full_context`.
- `memory_enabled` reduced estimated cost by `99.6`% versus `full_context`.
- `memory_enabled` reduced latency by `58.4`% versus `full_context`.

## Key Findings

- memory_enabled achieved a 171/300 pass rate (57.0%), versus 72/300 (24.0%) for full_context and 4/300 (1.3%) for short_context.
- memory_enabled used 349.4 prompt tokens on average, compared with 160543.9 for full_context.
- memory_enabled was 58.4% faster on average than full_context.
- full_context cost about 242.5x more per example than memory_enabled.
- Under the current must-not-include rules, stale/forbidden-fact violation rates were 4.0% for memory_enabled, 15.0% for full_context, and 4.0% for short_context.

## Recommended Next Steps

- Tighten the memory-mode answer format so the model is pushed to cite all retrieved facts that satisfy the gold constraints.
- Run a second benchmark pass after prompt tuning to see whether memory_enabled can move from 58.3% into the 70-80% range.
- Add a public-facing chart image or screenshot to the repo README for easier sharing.

## Sample Size Guidance

- The current run covers `300` examples total.
- Recommended minimum for a public directional benchmark: `100` examples total, or `20` per scenario family.
- Recommended stronger target for tighter 95% pass-rate intervals: `385` examples total.
- Rationale: Around 100 binary-scored examples gives a very rough 95% margin of error of about +/-10 percentage points at worst-case accuracy. Around 385 examples brings that down to about +/-5 percentage points.
