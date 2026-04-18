# MindCheck

> Are you using AI as a tool — or becoming dependent on it?

MindCheck analyses your AI conversation logs (Claude Code, Cursor, Codex, Gemini CLI) and measures your **cognitive engagement** over time. Not how much you use AI, but *how* you use it.

---

## The problem

AI tools are powerful. But there's a risk: the easier it gets to offload thinking, the less thinking you do. You might not notice it happening — until one day you can't solve problems without asking AI first.

MindCheck gives you a mirror.

---

## What it measures

| Signal | What it detects |
|---|---|
| **Question framing** | Do you form a hypothesis before asking, or just dump the problem? |
| **Agency** | Are you driving the conversation, or just reacting to AI output? |
| **Learning trajectory** | Are your questions getting deeper or simpler over time? |
| **Critical engagement** | Do you push back on AI answers, or accept everything? |
| **Self-reliance** | Do you attempt problems before asking for help? |
| **Topic retention** | Do you keep asking the same questions, or do they stick? |
| **Metacognition** | Do you reflect on your own approach and blind spots? |

---

## How it works

Three-tier signal extraction — designed to be cheap and accurate:

```
Tier 1: Structural rules    (free, offline)  — ratios, counts, timestamps
Tier 2: Semantic embeddings (free, offline)  — meaning, not keywords
Tier 3: LLM classification  (~$0.01/month)  — ambiguous edge cases only
```

Only your messages are analysed — AI responses are discarded. Results are cached locally so each new session costs fractions of a cent.

---

## Install

```bash
pip install mindcheck
```

Or from source:

```bash
git clone https://github.com/PatrickSqx/-MindCheck.git
cd -MindCheck
pip install -e ".[dev]"
```

---

## Usage

```bash
# Analyse all sessions in a folder
mindcheck analyze ./sessions/

# Generate a report for the last 30 days
mindcheck report --last 30d

# Score a single session file
mindcheck score session.jsonl

# Auto-discover sessions from known AI tool directories
mindcheck scan
```

---

## Sample output

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

## Supported formats

| Tool | Format | Auto-discovered |
|---|---|---|
| Claude Code | `.jsonl` | ✅ |
| Cursor | SQLite | ✅ |
| Codex CLI | `.jsonl` | ✅ |
| Gemini CLI | `.json` | ✅ |

---

## Privacy

Everything runs locally. No data leaves your machine unless you explicitly enable Tier 3 LLM classification with your own API key. Even then, only your short messages are sent — never AI responses, never full sessions.

---

## License

MIT
