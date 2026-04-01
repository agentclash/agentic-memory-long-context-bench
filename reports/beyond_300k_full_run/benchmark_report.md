# agentic-memory Beyond-300k Benchmark Report

On >300k-token conversations, agentic-memory retrieval outperformed both short-context and full-context baselines while using dramatically fewer prompt tokens than transcript stuffing.

## Benchmark Setup

- Dataset: `12` examples from the `beyond_250k` tier
- Transcript size: approximately `301143` to `318090` estimated tokens per example
- Model under test: `gemini-2.5-flash`
- Judge model: `gemini-2.5-flash-lite`
- Total mode runs: `36`

## Results

| Mode | Pass Rate | Avg Judge | Avg Latency (ms) | Avg Prompt Tokens | Avg Cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_enabled` | 7/12 (58.3%) | 0.8333 | 3470.32 | 289.2 | $0.000323 |
| `full_context` | 3/12 (25.0%) | 0.4417 | 6952.03 | 162590.2 | $0.049154 |
| `short_context` | 0/12 (0.0%) | 0.3500 | 3800.70 | 1647.8 | $0.000827 |

## Confidence Intervals

| Mode | 95% Wilson CI For Pass Rate |
| --- | ---: |
| `memory_enabled` | 31.9% to 80.7% |
| `full_context` | 8.9% to 53.2% |
| `short_context` | 0.0% to 24.2% |

## Why This Matters

- `memory_enabled` improved pass rate by `33.3` percentage points over `full_context`.
- `memory_enabled` reduced prompt-token volume by `99.8`% versus `full_context`.
- `memory_enabled` reduced estimated cost by `99.3`% versus `full_context`.
- `memory_enabled` reduced latency by `50.1`% versus `full_context`.

## Key Findings

- memory_enabled achieved a 7/12 pass rate (58.3%), versus 3/12 (25.0%) for full_context and 0/12 (0.0%) for short_context.
- memory_enabled used 289.2 prompt tokens on average, compared with 162590.2 for full_context.
- memory_enabled was 50.1% faster on average than full_context.
- full_context cost about 152.2x more per example than memory_enabled.
- All three modes recorded a 0.0% hallucination rate under the current must-not-include rules, so the largest separation came from answer completeness and relevance.

## Recommended Next Steps

- Tighten the memory-mode answer format so the model is pushed to cite all retrieved facts that satisfy the gold constraints.
- Run a second benchmark pass after prompt tuning to see whether memory_enabled can move from 58.3% into the 70-80% range.
- Add a public-facing chart image or screenshot to the repo README for easier sharing.

## Sample Size Guidance

- The current run is a pilot: `12` examples total.
- Recommended minimum for a public directional benchmark: `100` examples total, or `20` per scenario family.
- Recommended stronger target for tighter 95% pass-rate intervals: `385` examples total.
- Rationale: Around 100 binary-scored examples gives a very rough 95% margin of error of about +/-10 percentage points at worst-case accuracy. Around 385 examples brings that down to about +/-5 percentage points.
