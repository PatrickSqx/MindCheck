# MindCheck

> Are you using AI as a tool — or becoming dependent on it?

MindCheck analyses your AI conversation logs and measures your **cognitive engagement** — not how much you use AI, but *how* you use it.

---

## The problem

AI tools are powerful. But there's a risk: the easier it gets to offload thinking, the less thinking you do. You might not notice it happening — until one day you can't solve problems without asking AI first.

MindCheck gives you a mirror.

---

## What it measures

| Signal | What it detects |
|---|---|
| **Hypothesis level** | Do you form a hypothesis before asking, or just dump the problem? (0–4 scale) |
| **Agency** | Are you driving the conversation, or just reacting to AI output? |
| **Critical engagement** | Do you push back on AI answers, or accept everything? |
| **Self-reliance** | Do you attempt problems before asking for help? |
| **Metacognition** | Do you reflect on your own approach and blind spots? |
| **Delegation** | How often are you handing off thinking entirely? |

### Score bands

| Score | Meaning |
|---|---|
| 70–100 | Strong engagement — driving, hypothesising, thinking critically |
| 50–69 | Moderate — solid in places, room to push deeper before asking |
| 30–49 | Passive — leaning on AI for direction more than thinking first |
| 0–29 | Heavy delegation — most asks hand off the thinking entirely |

---

## How it works

Three-tier signal extraction — designed to be cheap and private:

```
Tier 1: Structural rules    (free, offline)  — ratios, counts, patterns
Tier 2: Semantic embeddings (free, offline)  — meaning, not just keywords
Tier 3: LLM classification  (~$0.01/month)  — ambiguous edge cases only
```

Only your messages are analysed — AI responses are discarded. Results are cached locally so re-running is instant.

---

## Install

```bash
pip install mindcheck
```

Or from source:

```bash
git clone https://github.com/PatrickSqx/MindCheck.git
cd MindCheck
pip install -e .
```

> **First run:** Tier 2 downloads a ~118 MB multilingual embedding model automatically. This only happens once.

---

## Usage

```bash
# Score a single session file
mindcheck score session.jsonl

# Score with Tier 3 LLM refinement
mindcheck score session.jsonl --tier 3

# Analyse all sessions in a folder
mindcheck analyze ./sessions/

# Auto-discover and report on last 30 days
mindcheck report --last 30d

# Show all auto-discovered session directories on this machine
mindcheck scan

# Cache management
mindcheck cache          # show cache stats
mindcheck cache --clear  # clear all cached scores
```

---

## Tier 3 setup (optional)

Tier 3 uses a cheap LLM to resolve messages that Tier 2 was uncertain about. It's optional — Tier 2 handles most sessions well on its own.

```bash
# Anthropic (key auto-detected from sk-ant- prefix)
mindcheck config --key sk-ant-xxxx

# OpenAI (key auto-detected from sk- prefix)
mindcheck config --key sk-xxxx

# Gemini (key auto-detected from AIza prefix)
mindcheck config --key AIzaxxxx

# Local Ollama (free, no key needed)
mindcheck config --provider ollama

# Choose a specific model
mindcheck config --model gemini-2.5-flash-lite

# Show current config and available models
mindcheck config --show
```

---

## Supported formats

| Tool | Auto-discovered |
|---|---|
| Claude Code | ✅ `~/.claude/projects/` |
| Cursor | ✅ `~/.cursor/projects/` |
| Codex CLI | ✅ `~/.codex/sessions/` |
| Gemini CLI | ✅ `~/.gemini/tmp/` |

Agent/subagent sessions are automatically filtered — only human conversations are scored.

---

## Privacy

Everything runs locally. No data leaves your machine unless you explicitly enable Tier 3 with your own API key. Even then, only short individual messages are sent — never AI responses, never full sessions.

---

## Roadmap

- **v1.0** — Tier 1/2/3 scoring, four parsers, SQLite cache, multilingual support
- **v1.1** — ChatGPT export parser, cross-session learning trajectory, prototype self-improvement loop

---

## License

MIT
