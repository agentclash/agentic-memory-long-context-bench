# Claim Boundaries

This file captures what the completed `n=300` benchmark supports, what it does not support, and what remains unanswered.

## What You Can Confidently Claim

1. Memory retrieval beats transcript stuffing on recall-heavy long-context tasks.
   Overall, `memory_enabled` reached `57.0%` versus `24.0%` for `full_context`, a gap of `33.0` percentage points.

2. Contradiction resolution is the strongest result in the benchmark.
   `memory_enabled` solved `60/60` (`100.0%`) while `full_context` solved `15/60` (`25.0%`).

3. The economics are unambiguous.
   `memory_enabled` used `349.4` prompt tokens versus `160543.9` for `full_context`, and cost about `244x` less per row.

4. Profile recall works.
   `memory_enabled` solved `42/60` (`70.0%`) while `full_context` solved `0/60` (`0.0%`).

5. Short context is effectively useless for these tasks.
   `short_context` finished at `1.3%` overall.

6. The benchmark infrastructure works.
   The run completed `900/900` rows with deterministic generation, checkpointing, and row-level cost/latency tracking.

## What You Cannot Claim

1. "Memory reduces hallucination."
   The current failure label is really stale/forbidden-fact citation under strict `must_not_include` rules, not a clean fabrication metric.

2. "Memory beats full context across all task types."
   It does not. In troubleshooting continuity, `full_context` scored `57/60` (`95.0%`) while `memory_enabled` scored `50/60` (`83.3%`).

3. "Procedural memory is a strong differentiator."
   It helps, but it is not solved. On procedure reuse, `memory_enabled` scored `18/60` (`30.0%`) while `full_context` scored `0/60`.

4. "This generalizes to production."
   The benchmark memory path still benefits from benchmark turn-kind labels at ingestion time.

5. "This works across models."
   The completed run used a single participant model family: `gemini-2.5-flash-lite`.

## What Is Genuinely Unanswered

1. How much perfect ingestion labeling inflates the result.
   This is still the biggest open question.

2. Why mixed long-context collapses.
   `memory_enabled` scored only `1/60` (`1.7%`) there.

3. Whether the typed memory taxonomy is itself important.
   We still need ablations to know whether semantic / episodic / procedural separation matters.

4. How this compares to actual memory products such as Mem0, Supermemory, or Zep.
   The benchmark compares against no-memory baselines, not competitor systems.

5. Whether the vocabulary and distractors are broad enough to support the widest generalization claims.

## Summary

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
