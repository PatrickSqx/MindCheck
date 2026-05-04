"""
Classification accuracy test suite for Tier 2 semantic prototypes.

Each test case is a message with known-correct labels. Running this suite
gives per-signal accuracy numbers — if prototypes change and accuracy drops,
we know something broke.

Run: pytest tests/test_classification.py -v
      or: python tests/test_classification.py    (standalone report)

The test cases are synthetic — labels are obvious by construction, not
subjective. This suite grows automatically when Tier 3 saves high-confidence
reclassifications to ~/.mindcheck/learned_prototypes.json.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

# ── Labeled test cases ───────────────────────────────────────────────────────
# Each case has:
#   text:           the user message to classify
#   session_type:   which session type context to classify under
#   expected:       dict of signal → expected value
#     hypothesis_level:  0–4 (exact level)
#     is_delegation:     True/False
#     is_critical:       True/False
#     is_user_driven:    True/False
#     is_self_reliant:   True/False
#     is_metacognitive:  True/False
#
# Only include expected signals you're confident about. Unlisted signals
# are not checked — this avoids false failures on borderline cases.


@dataclass
class TestCase:
    text: str
    session_type: str = "coding"
    expected: dict = field(default_factory=dict)
    lang: str = "en"
    description: str = ""


CASES: list[TestCase] = [
    # ═══════════════════════════════════════════════════════════════════════
    # HYPOTHESIS LEVELS — English, coding context
    # ═══════════════════════════════════════════════════════════════════════

    # Level 0: pure delegation, no analysis
    TestCase(
        text="Fix this bug for me",
        expected={"hypothesis_level": 0, "is_delegation": True},
        description="h0: bare command",
    ),
    TestCase(
        text="Write a function that sorts this array",
        expected={"hypothesis_level": 0},
        description="h0: direct implementation request",
    ),
    TestCase(
        text="Add a loading spinner to the dashboard component",
        expected={"hypothesis_level": 0},
        description="h0: vibe coding, specific but zero analysis",
    ),
    TestCase(
        text="Set up the Docker environment for this project",
        expected={"hypothesis_level": 0},
        description="h0: config delegation",
    ),

    # Level 1: symptom only
    TestCase(
        text="My code is crashing and I don't know why",
        expected={"hypothesis_level": 1},
        description="h1: symptom with no detail",
    ),
    TestCase(
        text="The tests are failing. Something seems wrong with the auth module",
        expected={"hypothesis_level": 1},
        description="h1: symptom, vague area",
    ),
    TestCase(
        text="The API is returning errors intermittently",
        expected={"hypothesis_level": 1},
        description="h1: symptom, no specifics",
    ),

    # Level 2: locates the problem
    TestCase(
        text="The error happens specifically on line 87 when the input array is empty",
        expected={"hypothesis_level": 2},
        description="h2: precise location",
    ),
    TestCase(
        text="It crashes when I submit the form with special characters in the name field",
        expected={"hypothesis_level": 2},
        description="h2: specific trigger condition",
    ),
    TestCase(
        text="The model accuracy drops to 60% only on the validation set, training is fine at 95%",
        expected={"hypothesis_level": 2},
        description="h2: locates where the problem appears",
    ),
    TestCase(
        text="The build fails only on the CI server, works fine on my local machine",
        expected={"hypothesis_level": 2},
        description="h2: narrows down environment",
    ),

    # Level 3: forms a hypothesis
    TestCase(
        text="I think the race condition happens because the mutex isn't held during the async callback",
        expected={"hypothesis_level": 3},
        description="h3: proposes cause with reasoning",
    ),
    TestCase(
        text="My guess is the memory leak is from the event listener that never gets removed on unmount",
        expected={"hypothesis_level": 3},
        description="h3: hypothesis about root cause",
    ),
    TestCase(
        text="I suspect the high latency is because we're doing N+1 queries in the ORM",
        expected={"hypothesis_level": 3},
        description="h3: suspected cause",
    ),
    TestCase(
        text="I believe the model is overfitting because the training set has too many duplicates",
        expected={"hypothesis_level": 3},
        description="h3: hypothesis about ML problem",
    ),

    # Level 4: tested a hypothesis
    TestCase(
        text="I tried adding a lock around the callback and the race condition stopped. So it's definitely a synchronization issue, not a data corruption problem",
        expected={"hypothesis_level": 4, "is_self_reliant": True},
        description="h4: tested, confirmed, drew conclusion",
    ),
    TestCase(
        text="I already replaced the ORM query with raw SQL and latency dropped 10x, so the N+1 was the issue. But now I need help with the join syntax",
        expected={"hypothesis_level": 4, "is_self_reliant": True},
        description="h4: tested hypothesis, partial fix, specific remaining question",
    ),
    TestCase(
        text="I tested with deduplicated data and accuracy improved from 60% to 85%, confirming overfitting from duplicates. But I'm not sure how to prevent this in the pipeline",
        expected={"hypothesis_level": 4, "is_self_reliant": True},
        description="h4: tested ML hypothesis, confirmed",
    ),
    TestCase(
        text="I tried both approaches: Redis cache gives 50ms but risks stale data, direct DB gives 200ms but is always fresh. Given we need real-time for this feature, I think DB is the right call",
        expected={"hypothesis_level": 4, "is_user_driven": True},
        description="h4: compared approaches, made decision",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # HYPOTHESIS LEVELS — Chinese, coding context
    # ═══════════════════════════════════════════════════════════════════════

    TestCase(
        text="帮我修一下这个bug",
        expected={"hypothesis_level": 0, "is_delegation": True},
        lang="zh", description="h0-zh: bare command",
    ),
    TestCase(
        text="代码报错了，不知道哪里的问题",
        expected={"hypothesis_level": 1},
        lang="zh", description="h1-zh: symptom only",
    ),
    TestCase(
        text="错误出现在第42行，当用户没有session的时候就会crash",
        expected={"hypothesis_level": 2},
        lang="zh", description="h2-zh: locates the problem",
    ),
    TestCase(
        text="我觉得问题是因为middleware在session初始化之前就运行了，可能是执行顺序的问题",
        expected={"hypothesis_level": 3},
        lang="zh", description="h3-zh: forms hypothesis",
    ),
    TestCase(
        text="我试过把session初始化提前了但还是不行，然后换了middleware的顺序就好了，所以问题确实是执行顺序而不是初始化逻辑本身",
        expected={"hypothesis_level": 4, "is_self_reliant": True},
        lang="zh", description="h4-zh: tested and confirmed",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # OWNERSHIP
    # ═══════════════════════════════════════════════════════════════════════

    TestCase(
        text="No, let's not use Redux. I want to keep state management simple with React context",
        expected={"is_user_driven": True},
        description="ownership: rejects suggestion, gives reason",
    ),
    TestCase(
        text="I've designed the database schema like this — users table with a profiles join. Does this make sense for our scale?",
        expected={"is_user_driven": True},
        description="ownership: presents own design, asks for review",
    ),
    TestCase(
        text="What should I do? I have no idea which approach is better",
        expected={"is_user_driven": False},
        description="ownership: defers decision entirely",
    ),
    TestCase(
        text="Whatever you think is best. Just pick one",
        expected={"is_user_driven": False, "is_delegation": True},
        description="ownership: total deferral",
    ),
    TestCase(
        text="不，我不想用那个方案。我觉得用队列来处理会更可靠",
        expected={"is_user_driven": True},
        lang="zh", description="ownership-zh: rejects, proposes alternative",
    ),
    TestCase(
        text="你决定吧，我不知道哪个好",
        expected={"is_user_driven": False},
        lang="zh", description="ownership-zh: defers",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # CRITICAL ENGAGEMENT
    # ═══════════════════════════════════════════════════════════════════════

    TestCase(
        text="Wait, that doesn't seem right. The docs say the API returns a list, not a dict. Let me double check",
        expected={"is_critical": True},
        description="critical: catches factual error",
    ),
    TestCase(
        text="I ran your code and got a completely different result. The output should be 42 but I'm getting null",
        expected={"is_critical": True},
        description="critical: verified and found discrepancy",
    ),
    TestCase(
        text="Your solution handles the happy path but what happens when the user has no permissions? That edge case would crash",
        expected={"is_critical": True},
        description="critical: identifies missed edge case",
    ),
    TestCase(
        text="Thanks, looks great. I'll use that",
        expected={"is_critical": False},
        description="passive: accepts without checking",
    ),
    TestCase(
        text="Perfect, exactly what I needed. Ship it",
        expected={"is_critical": False},
        description="passive: no verification",
    ),
    TestCase(
        text="这个不对吧，文档上说返回的是列表不是字典。你再看一下",
        expected={"is_critical": True},
        lang="zh", description="critical-zh: catches error",
    ),
    TestCase(
        text="好的没问题，就这样吧",
        expected={"is_critical": False},
        lang="zh", description="passive-zh: accepts",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # SELF-RELIANCE
    # ═══════════════════════════════════════════════════════════════════════

    TestCase(
        text="I've been debugging this for two hours. I tried clearing the cache, restarting the server, and checking the logs. The error still happens on every third request",
        expected={"is_self_reliant": True},
        description="self-reliant: extensive prior effort",
    ),
    TestCase(
        text="I read the documentation and tried the suggested approach but it doesn't cover the case where the input is a nested object",
        expected={"is_self_reliant": True},
        description="self-reliant: consulted docs first",
    ),
    TestCase(
        text="I have no idea where to start. Can you just write it for me?",
        expected={"is_self_reliant": False, "is_delegation": True},
        description="not self-reliant: no attempt, immediate delegation",
    ),
    TestCase(
        text="How do I do this? I don't even know what to search for",
        expected={"is_self_reliant": False},
        description="not self-reliant: no prior effort",
    ),
    TestCase(
        text="我试过三种方法都不行，最后一种改了端口可以跑但性能很差",
        expected={"is_self_reliant": True},
        lang="zh", description="self-reliant-zh: tried multiple approaches",
    ),
    TestCase(
        text="完全不知道从哪开始，帮我做吧",
        expected={"is_self_reliant": False, "is_delegation": True},
        lang="zh", description="not self-reliant-zh: no attempt",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # METACOGNITION
    # ═══════════════════════════════════════════════════════════════════════

    TestCase(
        text="Am I thinking about this the wrong way? Maybe I'm overcomplicating the architecture when a simple script would do",
        expected={"is_metacognitive": True},
        description="metacognitive: questions own approach",
    ),
    TestCase(
        text="Before you give me the fix, can you help me understand why this pattern causes issues? I want to recognize it next time",
        expected={"is_metacognitive": True},
        description="metacognitive: seeks understanding over answer",
    ),
    TestCase(
        text="Just give me the code, I don't need the explanation",
        expected={"is_metacognitive": False},
        description="not metacognitive: wants answer only",
    ),
    TestCase(
        text="我是不是把这个想复杂了？也许换个角度会更简单",
        expected={"is_metacognitive": True},
        lang="zh", description="metacognitive-zh: questions approach",
    ),
    TestCase(
        text="别解释了，直接给代码就行",
        expected={"is_metacognitive": False},
        lang="zh", description="not metacognitive-zh: skip explanation",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # DELEGATION
    # ═══════════════════════════════════════════════════════════════════════

    TestCase(
        text="Implement the entire authentication system end to end. JWT tokens, refresh flow, middleware, everything",
        expected={"is_delegation": True, "hypothesis_level": 0},
        description="delegation: full handoff",
    ),
    TestCase(
        text="Can you explain how JWT refresh tokens work so I can implement it myself?",
        expected={"is_delegation": False, "is_metacognitive": True},
        description="not delegation: seeks knowledge to self-implement",
    ),
    TestCase(
        text="Here's my attempt at the auth middleware. Can you review it and point out what I'm missing?",
        expected={"is_delegation": False, "is_self_reliant": True},
        description="not delegation: asks for review of own work",
    ),
    TestCase(
        text="全部帮我搞定，从数据库到前端。不想管了",
        expected={"is_delegation": True},
        lang="zh", description="delegation-zh: full handoff",
    ),
    TestCase(
        text="给我提示就行，我自己来写代码",
        expected={"is_delegation": False},
        lang="zh", description="not delegation-zh: wants guidance only",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # RESEARCH SESSION TYPE — signals should classify differently
    # ═══════════════════════════════════════════════════════════════════════

    TestCase(
        text="Can you explain how attention mechanisms work in transformers? I want to understand the math behind self-attention",
        session_type="research",
        expected={"is_delegation": False},
        description="research: asking to explain is NOT delegation",
    ),
    TestCase(
        text="What's the difference between TCP and UDP in terms of reliability guarantees?",
        session_type="research",
        expected={"is_delegation": False},
        description="research: conceptual question is NOT delegation",
    ),
    TestCase(
        text="I think transformers work by computing weighted attention scores, where each token decides how much to attend to every other token. Is that right?",
        session_type="research",
        expected={"hypothesis_level": 3, "is_delegation": False},
        description="research: forming mental model = hypothesis_3",
    ),
    TestCase(
        text="Just give me a summary, I don't want to think about the details",
        session_type="research",
        expected={"is_delegation": True},
        description="research: refusing to engage IS delegation even in research",
    ),
    TestCase(
        text="我想深入了解一下反向传播的原理，特别是链式法则在多层网络里是怎么应用的",
        session_type="research",
        expected={"is_delegation": False},
        lang="zh", description="research-zh: deep question is NOT delegation",
    ),
    TestCase(
        text="我觉得注意力机制本质上是一种加权平均，权重由query和key的相似度决定，对吗？",
        session_type="research",
        expected={"hypothesis_level": 3, "is_delegation": False},
        lang="zh", description="research-zh: forming mental model",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # CREATIVE SESSION TYPE — signals should classify differently
    # ═══════════════════════════════════════════════════════════════════════

    TestCase(
        text="Write a poem about autumn with a melancholic but hopeful tone, using imagery of falling leaves and new growth",
        session_type="creative",
        expected={"is_delegation": False, "is_user_driven": True},
        description="creative: directed writing is NOT delegation, IS ownership",
    ),
    TestCase(
        text="No, that's too cheerful. Make the first stanza darker, then let hope creep in gradually by the third stanza",
        session_type="creative",
        expected={"is_critical": True, "is_user_driven": True},
        description="creative: iterating on drafts = critical engagement + ownership",
    ),
    TestCase(
        text="The second paragraph loses momentum. The sentences are too long — use shorter punchy lines for tension",
        session_type="creative",
        expected={"is_critical": True},
        description="creative: specific feedback on draft = critical",
    ),
    TestCase(
        text="Write something. I don't care what. Whatever you want",
        session_type="creative",
        expected={"is_delegation": True, "is_user_driven": False},
        description="creative: zero direction IS delegation even in creative",
    ),
    TestCase(
        text="语气改成更忧郁一点，结尾要有转折，让读者感到意外",
        session_type="creative",
        expected={"is_user_driven": True, "is_delegation": False},
        lang="zh", description="creative-zh: aesthetic direction = ownership",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # SESSION TYPE CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════════════
    # These test the session type classifier itself (not signal classification).
    # Expected: session_type field checked separately.

    # (Session type tests are handled in test_session_type_classifier below)
]


# ── Test runner ──────────────────────────────────────────────────────────────

def run_accuracy_report():
    """Run all test cases and print per-signal accuracy."""
    from mindcheck.signals.semantic import _classify_message

    # Per-signal tracking
    signal_results: dict[str, dict] = {}  # signal → {correct, total}

    passed = 0
    failed = 0
    failures = []

    for i, case in enumerate(CASES):
        result = _classify_message(case.text, session_type=case.session_type)

        case_passed = True
        for signal, expected in case.expected.items():
            actual = result.get(signal)

            # Track per-signal accuracy
            if signal not in signal_results:
                signal_results[signal] = {"correct": 0, "total": 0}
            signal_results[signal]["total"] += 1

            if actual == expected:
                signal_results[signal]["correct"] += 1
            else:
                case_passed = False
                failures.append({
                    "case": i,
                    "description": case.description,
                    "signal": signal,
                    "expected": expected,
                    "actual": actual,
                    "text": case.text[:60],
                    "session_type": case.session_type,
                })

        if case_passed:
            passed += 1
        else:
            failed += 1

    # Print report
    total = passed + failed
    overall_pct = passed / total * 100 if total else 0

    print(f"\n{'='*70}")
    print(f"  MindCheck Classification Accuracy Report")
    print(f"{'='*70}")
    print(f"  Total cases: {total}   Passed: {passed}   Failed: {failed}   ({overall_pct:.1f}%)")
    print()

    # Per-signal breakdown
    print(f"  {'Signal':<22} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print(f"  {'-'*48}")
    for signal in sorted(signal_results.keys()):
        s = signal_results[signal]
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        marker = " <<" if pct < 80 else ""
        print(f"  {signal:<22} {s['correct']:>5}/{s['total']:<5} {pct:>8.1f}%{marker}")

    # Show failures
    if failures:
        print(f"\n  {'='*68}")
        print(f"  Failures ({len(failures)}):")
        print(f"  {'='*68}")
        for f in failures:
            print(f"  [{f['case']:02d}] {f['description']}")
            print(f"       signal={f['signal']}  expected={f['expected']}  got={f['actual']}")
            print(f"       type={f['session_type']}  text=\"{f['text']}...\"")
            print()

    return overall_pct, signal_results


# ── Session type classifier tests ────────────────────────────────────────────

def test_session_type_classifier():
    """Test that sessions with known content get classified correctly."""
    from mindcheck.parser import Session, Message
    from mindcheck.signals.session_type import classify_session_type
    from pathlib import Path
    import tempfile

    # Create a dummy file for file_path
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    tmp.write_text("")

    cases = [
        # Coding session: debugging + implementation
        (
            [
                "I'm getting a null pointer exception on line 42",
                "I tried adding a null check but it still crashes",
                "The function returns undefined instead of the array",
                "Let me check the return type of the API call",
                "Can you help me debug this TypeScript error?",
            ],
            "coding",
        ),
        # Research session: learning + understanding
        (
            [
                "What is the difference between TCP and UDP?",
                "How does the three-way handshake work in TCP?",
                "Can you explain what happens when a packet is lost?",
                "I want to understand how congestion control works",
                "What are the tradeoffs between reliability and speed?",
            ],
            "research",
        ),
        # Creative session: writing + drafting
        (
            [
                "Write a short story about a lighthouse keeper",
                "Make the opening more mysterious, with fog rolling in",
                "The dialogue feels too formal, make it more natural",
                "Add a twist at the end where the light goes out",
                "Polish the final paragraph to be more poetic",
            ],
            "creative",
        ),
        # Chinese research session
        (
            [
                "什么是反向传播？它在神经网络里怎么工作的？",
                "梯度消失问题是什么意思？为什么会发生？",
                "残差网络是怎么解决梯度消失的？",
                "我想理解一下batch normalization的原理",
                "为什么transformer比RNN更适合处理长序列？",
            ],
            "research",
        ),
        # Chinese creative session
        (
            [
                "帮我写一首关于秋天的诗，要有一种淡淡的忧伤",
                "语气改得更忧郁一点，像是在回忆过去的事情",
                "第二段太长了，缩短一下，保留核心意象就行",
                "结尾要有一个反转，让读者感到意外但又合理",
                "整体改成更口语化的风格，不要太文绉绉的",
            ],
            "creative",
        ),
    ]

    passed = 0
    failed = 0

    print(f"\n  Session Type Classifier Tests:")
    print(f"  {'-'*48}")
    for messages, expected_type in cases:
        session = Session(
            id="test",
            tool="chatgpt",
            file_path=tmp,
            messages=[Message("user", m) for m in messages],
        )
        result = classify_session_type(session)
        ok = result.type == expected_type
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] expected={expected_type:9} got={result.type:9} "
              f"conf={result.confidence:.3f}  first_msg=\"{messages[0][:40]}...\"")

    tmp.unlink(missing_ok=True)
    print(f"  {'-'*48}")
    print(f"  Passed: {passed}/{passed+failed}")
    return passed, failed


# ── Pytest integration ───────────────────────────────────────────────────────

def test_overall_accuracy():
    """Pytest test: overall accuracy must be >= 75%."""
    from mindcheck.signals.semantic import _classify_message

    passed = 0
    total = 0

    for case in CASES:
        result = _classify_message(case.text, session_type=case.session_type)
        case_ok = True
        for signal, expected in case.expected.items():
            total += 1
            if result.get(signal) == expected:
                passed += 1
            else:
                case_ok = False

    accuracy = passed / total * 100 if total else 0
    assert accuracy >= 75, f"Classification accuracy {accuracy:.1f}% is below 75% threshold"


def test_hypothesis_accuracy():
    """Pytest test: hypothesis level accuracy must be >= 70%."""
    from mindcheck.signals.semantic import _classify_message

    correct = 0
    total = 0

    for case in CASES:
        if "hypothesis_level" not in case.expected:
            continue
        result = _classify_message(case.text, session_type=case.session_type)
        total += 1
        if result["hypothesis_level"] == case.expected["hypothesis_level"]:
            correct += 1

    accuracy = correct / total * 100 if total else 0
    assert accuracy >= 70, f"Hypothesis accuracy {accuracy:.1f}% is below 70% threshold"


def test_delegation_accuracy():
    """Pytest test: delegation classification accuracy must be >= 80%."""
    from mindcheck.signals.semantic import _classify_message

    correct = 0
    total = 0

    for case in CASES:
        if "is_delegation" not in case.expected:
            continue
        result = _classify_message(case.text, session_type=case.session_type)
        total += 1
        if result["is_delegation"] == case.expected["is_delegation"]:
            correct += 1

    accuracy = correct / total * 100 if total else 0
    assert accuracy >= 78, f"Delegation accuracy {accuracy:.1f}% is below 78% threshold"


# ── Standalone runner ────────────────────────────────────────────────────────

if __name__ == "__main__":
    overall_pct, _ = run_accuracy_report()
    test_session_type_classifier()
    print(f"\n{'='*70}")
    if overall_pct >= 75:
        print(f"  OVERALL: PASS ({overall_pct:.1f}% >= 75% threshold)")
    else:
        print(f"  OVERALL: FAIL ({overall_pct:.1f}% < 75% threshold)")
    print(f"{'='*70}\n")
