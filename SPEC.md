# MindCheck — Project Specification

## What it is

A Python CLI tool that analyses AI conversation logs (JSONL/SQLite from Claude Code, Cursor, Codex CLI, Gemini CLI) and measures **cognitive engagement** over time. The goal is not to track how much someone uses AI, but *how* they use it — are they thinking for themselves, or outsourcing their thinking?

The core insight: the WALL-E risk isn't using AI, it's letting AI do the thinking and just executing the output. MindCheck detects whether the thinking is still happening.

---

## Problem Statement

People use AI tools daily but have no way to measure whether their engagement is healthy or atrophying. Unlike productivity trackers (which measure output) or usage dashboards (which measure volume), MindCheck measures the **quality of cognitive engagement** — hypothesis quality, agency, critical thinking, learning trajectory.

Nobody else is building this. It is a genuine market gap.

---

## Architecture — Three-Tier Signal Extraction

Designed to be cheap and accurate. Most signals don't need an LLM at all.

### Tier 1: Structural Rules (free, offline)
Pure math and regex. Language-agnostic. No API cost.

- `question_ratio` — fraction of user messages containing "?"
- `message_ratio` — user chars / AI chars (higher = more engaged)
- `turn_count` — number of exchanges
- `delegation_count` — "just fix/write/do/make" pattern
- `urgency_count` — "asap", "urgent", "quickly"
- `prior_attempt_count` — "I tried", "I tested", "I checked"
- `avg_user_length` — longer messages = more context = more engaged
- `long_message_ratio` — fraction of messages > 200 chars
- `single_line_ratio` — fraction of very short messages

### Tier 2: Semantic Embeddings (free, offline)
Uses `sentence-transformers` (all-MiniLM-L6-v2, 80MB, runs on CPU).
Handles phrasing variation that keywords miss.
Compares each user message to prototype embeddings via cosine similarity.

Signals detected:
- `hypothesis_level_avg` — 0 to 4 scale:
  - 0 = no attempt ("fix this")
  - 1 = symptom only ("there's an error")
  - 2 = locates problem ("fails on line 42")
  - 3 = forms hypothesis ("I think X because Y")
  - 4 = tested hypothesis ("I tried X, still fails, maybe Y")
- `agency_score` — fraction of turns where user drives direction (0–1)
- `critical_engagement` — fraction with pushback/verification (0–1)
- `self_reliance` — showed prior attempt before asking (0–1)
- `metacognition_score` — reflected on own approach (0–1)
- `delegation_penalty` — proportion of outsourcing language (0–1)

Prototype meanings are plain-text descriptions in `signals/semantic.py` — editing them improves accuracy without code changes.

### Tier 3: LLM Classification (cheap, ~$0.01/month)
Only called for messages where Tier 2 confidence is low.
Only user messages sent — never full sessions, never AI responses.
Supports Anthropic (Claude Haiku), OpenAI (GPT-4o-mini), Ollama (local, free).
Currently stubbed — wiring to Tier 2 confidence scores is next step.

---

## Composite Score (0–100)

| Category | Weight | Key signals |
|---|---|---|
| Question framing / hypothesis | 25% | hypothesis_level_avg |
| Agency | 20% | agency_score |
| Critical engagement | 20% | critical_engagement |
| Learning / self-reliance | 15% | self_reliance |
| Metacognition | 10% | metacognition_score |
| Structural ratios | 5% | question_ratio, turn_count, message_ratio |
| Delegation penalty | −20pts max | delegation_penalty |

---

## Signal Framework (full list from design session)

These are all the dimensions designed. Not all are implemented yet.

### Implemented
- Question framing (Tier 1 + 2)
- Hypothesis quality 0–4 (Tier 2)
- Agency / who drives (Tier 2)
- Critical engagement / pushback (Tier 2)
- Self-reliance / prior attempt (Tier 1 + 2)
- Metacognition (Tier 2)
- Delegation penalty (Tier 1 + 2)
- Structural ratios (Tier 1)

### Designed but not yet implemented
- **Learning trajectory** — complexity trend over time across sessions
- **Topic retention** — same topic repeated across sessions = not retaining
- **Teaching signals** — user explains something back to AI (highest engagement)
- **Sycophancy test** — does user push back on vague AI answers
- **Emotional/urgency pattern** — reactive vs deliberate mode over time
- **Scope of delegation** — large chunks vs targeted gaps
- **Cross-session improvement** — does question complexity increase?
- **The "I figured it out" signal** — user solved it themselves, just confirming

---

## Key Design Decisions

### Only analyse user messages
AI responses are discarded entirely. They're output, not signal. This cuts token usage by 75–80% for Tier 3.

### Incremental processing + SQLite cache
Process new sessions only, cache results. After initial batch, daily cost is ~$0.002.

### Cost reality
100 sessions × 20 user messages × 150 tokens = 300k tokens total.
At Claude Haiku pricing ($0.25/million) = $0.075 for entire history.
Tier 2 (embeddings) = $0.00.

### Embeddings over keywords
Keywords miss phrasing variation. "Go ahead and handle it" = delegation, but no keyword catches it.
Embeddings capture meaning — works across languages too.

### Prototype-driven classification
The `PROTOTYPES` dict in `signals/semantic.py` is plain English.
Improving the descriptions improves accuracy — no retraining, no code changes.

---

## File Structure

```
MindCheck/
├── mindcheck/
│   ├── __init__.py          version = "0.1.0"
│   ├── cli.py               CLI: analyze / score / report / scan commands
│   ├── parser.py            reads JSONL/JSON/SQLite, extracts user messages only
│   ├── scorer.py            aggregates signals → composite 0-100 score
│   ├── report.py            terminal (rich) + markdown output
│   └── signals/
│       ├── structural.py    Tier 1: pure math/regex, free, offline
│       ├── semantic.py      Tier 2: sentence-transformers embeddings, free, offline
│       └── llm.py           Tier 3: LLM classification stub, plugs into Anthropic/OpenAI/Ollama
├── tests/
│   ├── test_parser.py       parser tests (no API needed)
│   └── test_signals.py      Tier 1 structural signal tests (no API needed)
├── .gitignore
├── LICENSE                  MIT, Copyright 2026 PatrickSqx
├── README.md                full user-facing documentation
├── SPEC.md                  this file
└── pyproject.toml           hatchling build, `mindcheck` CLI entry point
```

---

## Supported Input Formats

| Tool | Format | Path (auto-discovered) |
|---|---|---|
| Claude Code | `.jsonl` | `~/.claude/projects/` |
| Cursor | SQLite `.db` | `AppData/Roaming/Cursor/User/workspaceStorage/` |
| Codex CLI | `.jsonl` | `~/.codex/` |
| Gemini CLI | `.json` | `~/.gemini/` or `~/.config/gemini/` |

Parser handles: role variations (user/human/assistant/ai/model), nested content arrays, timestamps in multiple formats (Unix ms, Unix s, ISO 8601).

---

## CLI Commands

```bash
mindcheck analyze ./sessions/          # analyse folder, output report.md
mindcheck score session.jsonl          # score single file, print to terminal
mindcheck report --last 30d            # auto-discover + report for time window
mindcheck scan                         # show known directories and their status
```

Options:
- `--tier 1|2|3` — max analysis tier (default 2, Tier 3 requires API key)
- `--output path` — output file for analyze command

---

## Sample Output

```
MindCheck Report — April 2026
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cognitive Engagement Score: 61/100  ↓ from 74 last month

WHAT CHANGED
  Hypothesis quality dropped from 58% → 31% of questions
  "Just do it" phrasing up 40% this month

PATTERN SPOTTED
  You've asked about async/await in 7 of the last 10 sessions.
  This topic isn't sticking — consider a focused read.

YOU'RE DOING WELL AT
  Critical engagement: caught AI mistakes 6 times ↑
  Strongest session Apr 14: you drove the full architecture discussion

NUDGE
  Before your next question, write one sentence about what
  you think is causing the problem. Even if you're wrong.
```

---

## Current Status

| Component | Status |
|---|---|
| Project scaffold | ✅ Done |
| GitHub repo | ✅ `github.com/PatrickSqx/-MindCheck` (private) |
| Parser (all 4 formats) | ✅ Written, needs real-world testing |
| Tier 1 structural signals | ✅ Implemented |
| Tier 2 semantic embeddings | ✅ Implemented, prototype definitions need tuning |
| Tier 3 LLM stub | ✅ Stubbed, needs wiring to Tier 2 confidence scores |
| Scorer | ✅ Implemented |
| Report (terminal + markdown) | ✅ Implemented |
| CLI | ✅ All 4 commands wired up |
| Tests | ✅ Parser + Tier 1 structural tests |
| Tier 2 tests | ❌ Need embedding tests (requires model download) |
| Tier 3 wiring | ❌ Not connected yet |
| Learning trajectory (cross-session) | ❌ Not implemented |
| Topic retention signal | ❌ Not implemented |
| Teaching signal | ❌ Not implemented |
| SQLite cache for incremental processing | ❌ Not implemented |
| Real-world validation on actual sessions | ❌ Not done yet |

---

## Immediate Next Steps

1. **Push scaffold to GitHub** — resolve merge conflicts, push clean state
2. **Install and test locally** — `pip install -e .` then `mindcheck scan`
3. **Run on real sessions** — validate parser reads actual Claude/Cursor JSONL
4. **Tune prototype definitions** — adjust `PROTOTYPES` in `semantic.py` based on real output
5. **Add SQLite cache** — so sessions aren't re-processed on every run
6. **Implement learning trajectory** — cross-session complexity trend
7. **Wire Tier 3** — connect LLM to low-confidence Tier 2 messages

---

## Relationship to Distill (sister project)

Distill (`C:\Users\26982\OneDrive\Desktop\JsonlToMD-v2\`) is a WPF desktop app (C# .NET 8) that converts AI sessions to Markdown and generates AI digests. It reads the same JSONL files.

MindCheck shares the parsing problem but has a completely different purpose:
- Distill → context continuity (compress sessions for reuse)
- MindCheck → cognitive reflection (measure engagement quality)

Parser logic could eventually be shared as a Python library. Keep as separate repos.

---

## Philosophy

> "Self-improving agent" = an agent that modifies its own instructions, memory, or code.
> The underlying model never changes. The *system around the model* improves.

MindCheck applies the same principle to humans. You can't change your brain's hardware.
But you can change the context — the habits, the questions you ask, the effort you put in before asking.
MindCheck measures whether that context is improving or degrading.
