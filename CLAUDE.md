# MindCheck — Project Context

## What this is
CLI tool that analyzes AI conversation logs and measures cognitive engagement.
Scores how users think when using AI, not just how much they use it.

## Architecture
- **Tier 1:** Structural rules (free, offline) — ratios, counts, patterns
- **Tier 2:** Semantic embeddings (free, offline) — cosine similarity against prototypes
- **Tier 3:** LLM classification (optional) — Gemini/OpenAI/Anthropic/Ollama for edge cases

## Key files
- `mindcheck/parser.py` — Session/Message parsing for Claude Code, Cursor, Codex, Gemini CLI, ChatGPT/Claude Chat imports
- `mindcheck/scorer.py` — SessionScore dataclass, composite scoring, score_session() pipeline
- `mindcheck/signals/structural.py` — Tier 1: question ratio, delegation count, message ratio
- `mindcheck/signals/semantic.py` — Tier 2: embedding-based classification (hypothesis, ownership, critical, delegation, etc.)
- `mindcheck/signals/subtext.py` — Subtext/authenticity: say-then-contradict, empty self-reliance, passive streaks, LLM intent analysis
- `mindcheck/signals/llm.py` — Tier 3: LLM reclassification of low-confidence messages
- `mindcheck/signals/session_type.py` — Auto-detect coding/research/creative/casual, per-type scoring weights
- `mindcheck/trajectory.py` — Cross-session learning trajectory, trend analysis, sparklines
- `mindcheck/report.py` — Rich terminal + markdown report output
- `mindcheck/cache.py` — SQLite cache keyed by (file_path, tier), auto-invalidates on CACHE_VERSION bump
- `mindcheck/cli.py` — Click CLI: analyze, score, report, import, trajectory, config, cache, scan

## Version history
- **v1.0** — Tier 1/2/3 scoring, four parsers, SQLite cache, multilingual (EN+ZH)
- **v1.1** — ChatGPT + Claude Chat import, session type detection, per-type scoring weights, classification test suite
- **v1.2** — Cross-session learning trajectory, trend analysis, trajectory persistence (~/.mindcheck/trajectory.json)
- **v1.3** — Subtext / illocutionary intent detection: authenticity scoring, say-then-contradict, empty self-reliance, passive acceptance streaks, hypothesis-without-followup, LLM performative/fake-curiosity detection. Rate-based penalty calibration. Integrated into scorer, report, cache.

## Design decisions
- Penalty calibration uses rates (pattern_count / total_messages), not absolute counts — long sessions were unfairly penalized with fixed penalties
- Authenticity modifier is a soft blend: `composite * (0.6 + 0.4 * auth)` so even low authenticity can't zero the score
- Subtext local detection is free (runs on Tier 2 classifications), LLM subtext only runs with Tier 3
- Cache version bump forces full re-analysis — current version is 10
- Classification test threshold is 75% (currently at 81.2%)
- Delegation prototype engineering: centroid-based cosine similarity is sensitive to shared vocabulary — genuinely borderline cases are a Tier 3 problem, not a prototype problem
- Python executable on this system is `py`, not `python`

## Next up (v1.4)
- Per-message subtext detail view: `--detail subtext` flag to show which messages were flagged and why
- SubtextSignals.flags already stores per-message flag data with indices, just needs display in report.py
- Show LLM surface-vs-intent analysis when Tier 3 ran (llm_intents has surface/subtext/pattern per message)
