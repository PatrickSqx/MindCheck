"""
Tier 2: Semantic signal extraction via local embeddings.
Uses sentence-transformers (runs offline, no API cost).
Handles phrasing variation — catches signals that keywords miss.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from mindcheck.parser import Session


@dataclass
class SemanticSignals:
    hypothesis_level_avg: float = 0.0    # 0–4 avg hypothesis quality
    ownership_score: float = 0.0            # 0–1 fraction of user-driven turns
    critical_engagement: float = 0.0     # 0–1 fraction with pushback / verification
    self_reliance: float = 0.0           # 0–1 showed prior attempt
    metacognition_score: float = 0.0     # 0–1 reflected on own approach
    delegation_penalty: float = 0.0      # 0–1 proportion of outsourcing language

    # Per-task-domain breakdown: {domain → {count, hypothesis_avg, delegation_rate}}
    # Domains: code, data, writing, research, planning, config
    task_breakdown: dict = field(default_factory=dict)

    # Per-message details for reporting
    per_message: list[dict] = field(default_factory=list)


# ── Prototype meanings ────────────────────────────────────────────────────────
# These are the "tuning knobs" — improve these descriptions to improve accuracy.
# Each prototype is a natural-language description of what the signal means,
# plus diverse examples to widen the semantic coverage.

PROTOTYPES: dict[str, list[str]] = {
    # Hypothesis levels (0–4)
    # Each level should cover: coding, data analysis, research/writing, general reasoning
    "hypothesis_0": [
        # Coding — direct commands
        "Fix this. Do this for me. Make it work. Just do it. Write the code.",
        "This is broken. It doesn't work. I have an error.",
        # Coding — vibe coding (specific but zero analysis)
        "Add a button that submits the form. Make the sidebar collapsible.",
        "Add error handling to this function. Make this responsive.",
        "Create a new component for the dashboard. Add a loading spinner.",
        "Change the color to blue. Move this div to the right. Make it bigger.",
        "Add a search bar. Implement dark mode. Add pagination to the list.",
        "Write a function that takes X and returns Y. Add input validation.",
        "Refactor this into smaller functions. Add types to everything.",
        # Data / analysis
        "Run the analysis. Generate the report. Clean this data for me.",
        "Process this dataset. Build the model. Create the chart.",
        # Research / writing
        "Write this section. Summarize this paper. Find sources for this claim.",
        "Draft the introduction. Rewrite this paragraph. Create the outline.",
        # Config / setup — direct setup commands
        "Set up Docker for me. Configure the CI pipeline. Install nginx.",
        "Write a Dockerfile. Set up the database. Create a Makefile.",
        "Deploy this to production. Set up the environment. Configure the server.",
        "Add authentication. Set up the API routes. Configure CORS.",
        "Install these dependencies. Set up linting. Configure the build.",
        # General
        "Handle this. Take care of it. Do this task for me. Just complete it.",
    ],
    "hypothesis_1": [
        # Coding
        "I'm getting a null pointer exception. There's an error in my code. It crashed.",
        "Something is wrong with the login. The tests are failing.",
        # Data / analysis
        "The chart looks wrong. The numbers don't add up. My results seem off.",
        "The model isn't performing well. The output isn't what I expected.",
        "Something is wrong with my analysis. The figures don't make sense.",
        # Research / writing
        "The argument doesn't flow well. The structure feels off.",
        "The explanation isn't clear. Something is missing from this section.",
        # Config / setup
        "The build is failing. Docker won't start. The deploy didn't work.",
        "Something's wrong with the server. The installation failed.",
        # General
        "This doesn't work correctly. The output is wrong. There's a problem somewhere.",
    ],
    "hypothesis_2": [
        # Coding
        "The error happens on line 42. It fails when the user has no session.",
        "It breaks when I submit the form with an empty field.",
        # Data / analysis
        "The outliers in column B are skewing the average.",
        "The model underperforms specifically on the test set, not training.",
        "It only fails when there are missing values in the date column.",
        "The issue appears in Q3 data but not Q1 — something changed mid-year.",
        # Research / writing
        "The argument breaks down in the third paragraph where I switch topics.",
        "The conclusion contradicts what I said in the introduction.",
        # Config / setup
        "The container crashes when it tries to bind port 80.",
        "The build fails only on the CI server, works fine locally.",
        "Permission denied on the /var/log directory specifically.",
        # General
        "The problem only happens under X condition. It works fine until Y occurs.",
        "It fails specifically when the input is large — small inputs are fine.",
    ],
    "hypothesis_3": [
        # Coding
        "I think the problem is that the session isn't initialized before the middleware runs.",
        "My guess is it's a timing issue with the async call. Could it be the promise isn't awaited?",
        "I suspect the null check is missing here. It seems like the config isn't loaded yet.",
        # Data / analysis
        "I suspect the model is overfitting — training accuracy is much higher than validation.",
        "My hypothesis is the correlation is spurious because X and Y are both driven by Z.",
        "I think the outliers are real signal, not noise — they cluster around a specific date.",
        "I believe the feature importance is misleading because the variables are collinear.",
        # Research / writing
        "I think the argument is weak because I'm assuming X without evidence.",
        "My guess is the reader loses track here because I haven't defined the key term yet.",
        "I suspect the structure is wrong — the conclusion should probably come earlier.",
        # Config / setup
        "I think the port conflict is because nginx is already running on 80.",
        "I suspect the env variable isn't being loaded because the .env file is outside the build context.",
        "My guess is the DNS isn't resolving because the container network is isolated.",
        # General
        "I think the root cause is X because Y only happens when Z is true.",
        "My hypothesis is that A is caused by B, not C, because the pattern matches B.",
    ],
    "hypothesis_4": [
        # Coding
        "I tried moving the session init earlier but it still fails. Maybe it's the order of middleware?",
        "I tested with a hardcoded value and it worked, so the issue must be in how I'm reading the env variable.",
        "I already checked the network tab and the request is correct, so it must be a server-side issue.",
        # Data / analysis
        "I tested with a smaller subset and the pattern holds, so it's not a sample size issue.",
        "I already tried normalizing the data — problem persists, so it might be the model architecture.",
        "I checked both approaches: A has better precision but worse recall. I think we should optimise for precision here because of X.",
        "I removed the outliers and re-ran — the correlation weakened, which confirms they were driving it.",
        # Research / writing
        "I tried restructuring the argument but the same objection applies. Maybe the premise itself is wrong.",
        "I already cut 500 words and it's still too long. I think the second section can be merged with the third.",
        # Config / setup
        "I tried changing the port to 8080 and it works, so something else is bound to 80.",
        "I already checked the firewall rules and they're fine, so it must be the Docker network config.",
        "I tested with a fresh container and it works — so the issue is in the cached image layer.",
        # General
        "I already ruled out X and Y by testing them separately. The only remaining explanation is Z.",
        "I tried both approaches — A is faster but B is more accurate. Given our constraints, I think B is right.",
    ],

    # Ownership (user-driven vs ai-driven)
    "user_driven": [
        # Steering direction
        "I want to understand how this works. Can you explain the tradeoffs?",
        "I've designed the architecture like this — does this make sense?",
        "I'm thinking about using X approach. What would you change?",
        "Here's what I've built so far. I need help with this specific part.",
        # Making decisions
        "Let's go with option A because it handles our scale better.",
        "I'd rather use Postgres here — SQLite won't cut it for concurrent writes.",
        "No, let's not do it that way. I want to keep it simple.",
        # Setting scope and constraints
        "Let's focus on the auth flow first, then tackle the UI later.",
        "I need this to handle at most 1000 concurrent users.",
        "Here's the context — we have a hard deadline and limited memory.",
        # Iterating with own ideas
        "What if we changed the retry logic to exponential backoff instead?",
        "I want to try a different approach — what about using a queue here?",
        # Casual / terse variants
        "nah let's do X instead",
        "I'd prefer Y actually",
    ],
    "ai_driven": [
        # Deferring decisions
        "What should I do next? Tell me what to build. What's the best approach?",
        "Just decide for me. You know best. Whatever you think is fine.",
        "I'll do whatever you suggest. What do you recommend?",
        # Passive acceptance
        "sure go ahead. ok do it. sounds good just do that.",
        "let's do what you said. yeah that works fine.",
        "whatever you think. up to you. your call.",
        # Asking AI to lead
        "What should I focus on first? Which one is better?",
        "Which framework should I pick? What technology should I use?",
        "How should I structure this? What's the right way?",
        # Lacking own perspective
        "I don't have a preference. Either way is fine with me.",
        "You pick. I'm not sure which is better.",
    ],

    # Critical engagement
    "critical": [
        # Explicit disagreement
        "Wait, that doesn't seem right because. I disagree — here's why.",
        "Are you sure about that? I thought it worked differently.",
        "That approach would break if X happens. You missed the edge case.",
        "I checked your solution and it has a bug. Let me explain.",
        # Questioning accuracy
        "Hmm that doesn't match what the docs say. Let me double check.",
        "I don't think that's correct — the API returns a list, not a dict.",
        "Are you hallucinating? That function doesn't exist in this library.",
        # Catching mistakes
        "You forgot to handle the null case. What happens when input is empty?",
        "This would fail on Windows — the path separator is different.",
        "wait no that's wrong lol. that output doesn't match what I see.",
        # Requesting justification
        "Why did you choose that approach over X? What's the reasoning?",
        "Can you justify that? I'm not convinced this is the right way.",
        # Verifying output
        "Let me test this first before we move on.",
        "I ran your code and got a different result. Something's off.",
    ],
    "passive": [
        # Accepting without checking
        "Thanks, looks good. That works, great. Perfect, I'll use that.",
        "OK I'll copy that. Great answer, thank you.",
        "Awesome, exactly what I needed. Perfect thanks.",
        "Got it, makes sense. I'll go with that.",
        "Looks right to me. Ship it. LGTM.",
        "Cool, moving on. Next thing.",
        "ok great. nice. thanks that helps.",
        "yep that works. good enough. done.",
        "I trust your judgment on this one.",
        "Sounds reasonable, let's go with it.",
    ],

    # Self-reliance
    "self_reliant": [
        # Showed prior effort — coding
        "I tried X but it didn't work. I already attempted Y. I've been debugging this for an hour.",
        "I figured it out but want to double-check. I solved it but curious if there's a better way.",
        "I've narrowed it down to these three lines. I think I know the issue.",
        # Showed prior effort — research
        "I read the docs but they don't cover this case.",
        "I've searched Stack Overflow and tried the top answers — none worked.",
        "I already looked into this and found two approaches, but I'm unsure which fits.",
        # Showed prior effort — casual/terse
        "been stuck on this for a while, tried restarting and clearing cache.",
        "I googled it but couldn't find anything relevant.",
        "already tried the obvious fix, didn't help.",
        # Partial solution
        "I got it mostly working, just stuck on the last part.",
        "Here's my current approach — it works for case A but not B.",
        "I wrote a first draft but something feels off about the structure.",
    ],
    "not_self_reliant": [
        # No attempt
        "I have no idea where to start. I don't know how to do this. Can you just write it?",
        "I give up. Can you fix it for me? I don't understand the error.",
        # Immediate helplessness
        "I'm completely lost. I don't even know what to search for.",
        "I've never done this before, can you walk me through everything?",
        "This is too hard. Can you just do it?",
        "I don't want to think about it, just handle it.",
        # Zero context provided
        "it's broken help. fix please. doesn't work.",
        "how do I do this? I have no clue.",
        "Can someone just solve this for me?",
        "I don't understand any of this.",
    ],

    # Metacognition
    "metacognitive": [
        # Questioning own approach
        "Am I approaching this the wrong way? What am I missing in my thinking?",
        "Am I thinking about this wrong? Maybe I'm overcomplicating the architecture.",
        "I'm not sure my mental model of this is right. Can you critique my approach?",
        "What blind spots might I have here? Is this the right way to think about it?",
        "I want to make sure I understand, not just copy the solution.",
        # Reflecting on process
        "I keep making the same mistake — what's the pattern I'm missing?",
        "I think I'm overcomplicating this. Am I overthinking it?",
        "Maybe I'm looking at this from the wrong angle entirely.",
        # Seeking understanding over answers
        "Before you give me the fix, can you help me understand why it breaks?",
        "I don't just want the answer, I want to know how to find it myself next time.",
        "Walk me through the reasoning so I can learn the pattern.",
        "Can you explain how this works so I can implement it myself?",
        "Help me understand the concept, not just the solution.",
        # Awareness of own gaps
        "I realize I don't fully understand how X works under the hood.",
        "I might be confused about the fundamentals here.",
        # Learning-oriented requests
        "I want to learn how to solve this type of problem, not just get the answer.",
        "Can you teach me the underlying principle so I recognize this pattern?",
    ],
    "not_metacognitive": [
        # Wants answer only
        "Just give me the answer. I don't need the explanation. Skip the details.",
        "Don't explain, just show me the code.",
        "Too long, just tell me what to do.",
        "I don't care why, just fix it.",
        "Skip the theory, give me the solution.",
        "Just the command please, no explanation needed.",
        "TLDR what do I type?",
        "Give me the short version.",
    ],

    # Delegation — outsourcing thinking/execution rather than engaging
    "delegation": [
        # Direct handoff
        "Just do it. Fix it for me. Write the whole thing. Go ahead and implement it.",
        "Can you just handle this? Do whatever you think is best. You decide.",
        "Build this feature. Write this function. Generate the code. Create the file.",
        "I need you to write this for me. Can you take care of this? Please implement.",
        "Just complete it. Finish the rest. Do the remaining parts.",
        # Terse delegation
        "do it. make it. build it. write it. fix it.",
        "go ahead. proceed. continue. keep going.",
        "implement the whole thing end to end.",
        # Outsourcing thinking
        "Figure out the best approach and just do it.",
        "You handle the design, I just need it done.",
        "Take care of everything — I don't want to think about this.",
        # "For me" pattern — key delegation signal
        "Fix this for me. Do this for me. Solve this for me.",
        "Handle this for me. Write this for me. Debug this for me.",
        # Short imperative + "for me" (matches test cases)
        "Fix this bug for me. Sort this out for me. Make this work for me.",
        "Solve this problem for me. Clean this up for me. Just handle it.",
        "Fix this bug for me, I don't want to debug it.",
        "There's a bug, just fix it for me. Fix the error for me.",
        "Can you fix this for me? I don't know what's wrong.",
        # Helplessness + handoff
        "I have no idea, just do it. I don't know, you figure it out.",
        "No idea where to start, just take care of it.",
        "I can't figure this out, just fix it for me. I give up, you do it.",
        "I don't know where to begin, can you just handle it for me?",
        "I'm lost, just do the whole thing. I have no clue, you take over.",
    ],
    "not_delegation": [
        # Collaborative engagement
        "I tried this approach and want to understand why it fails.",
        "Here's my attempt — what did I get wrong?",
        "Can you walk me through how this works? I want to solve it on my own.",
        "I want to understand the tradeoffs before we decide.",
        # Seeking guidance not execution
        "Can you point me in the right direction? I'll write the code myself.",
        "What should I look into? I want to learn how to solve this.",
        "Give me hints, don't give me the full answer.",
        "What do I need to learn to handle this myself?",
        "Help me think through this, not do it for me.",
        "Can you review my approach rather than rewriting it?",
        # Self-implementation intent — "I'll do it myself"
        "Just give me a hint, I'll code it myself.",
        "Point me in the right direction, I want to implement it on my own.",
        "I just need guidance, I'll write the actual code.",
    ],

    # ── Task domains ──────────────────────────────────────────────────────────
    # Used to classify what kind of task each message is about.
    # Higher similarity to a domain → message belongs to that domain.

    "task_code": [
        "Write a function to do X. Fix this bug. Implement this feature.",
        "I'm getting an error in my Python code. Debug this TypeScript.",
        "Refactor this code. Add unit tests. Review this pull request.",
        "How do I implement X in React? Help me write a SQL query.",
        "This class is broken. The API call is failing. The test won't pass.",
        "Add error handling. Optimise this loop. Why is this function slow?",
    ],
    "task_data": [
        "Analyse this dataset. Build a model to predict X. Train a classifier.",
        "Run a statistical test on this data. Create a visualization.",
        "Clean this CSV. Why is my pandas code slow? Optimise this query.",
        "The model isn't performing well. What features should I use?",
        "Plot this data. Calculate the correlation. Evaluate model accuracy.",
        "Feature engineering, data preprocessing, cross-validation, overfitting.",
    ],
    "task_writing": [
        "Write an introduction for my essay. Edit this paragraph.",
        "Summarize this article. Draft an email to my client.",
        "Rewrite this to be more concise. Improve the flow of this section.",
        "Write a cover letter. Create a report. Polish this text.",
        "The argument isn't clear. Help me structure this piece.",
        "Proofread this. Make it sound more professional. Shorten this.",
    ],
    "task_research": [
        "Explain how transformers work. What is X and how does it work?",
        "What's the difference between X and Y? How does Z work under the hood?",
        "Summarize this paper. What are the tradeoffs of approach X?",
        "I want to understand X better. Can you teach me about Y?",
        "What causes X? Why does Y happen? What's the best way to learn Z?",
        "Give me an overview of this topic. What does the research say?",
    ],
    "task_planning": [
        "Help me design the architecture for this system.",
        "How should I structure this project? What approach should I take?",
        "I need to plan a roadmap. What's the best design pattern here?",
        "Should I use X or Y for this? How should I organise the codebase?",
        "Think through the tradeoffs. Help me decide between these options.",
        "What are the risks? How do I prioritise? Plan out the next steps.",
    ],
    "task_config": [
        "How do I install X? Set up this environment. Configure this tool.",
        "My Docker container won't start. Deploy this to production.",
        "Set up CI/CD. Write a Makefile. Configure nginx. Set up the database.",
        "I can't get this package to install. The build is failing.",
        "Set up authentication. Configure environment variables. Write a Dockerfile.",
        "Permission denied. Port already in use. The server won't start.",
    ],
}

# ── Chinese prototypes (separate centroids, no dilution) ─────────────────────
# Keys use _zh suffix. Classification takes max(en, zh) similarity.

PROTOTYPES_ZH: dict[str, list[str]] = {
    "hypothesis_0_zh": [
        # Pure delegation — no analysis
        "帮我修一下。直接做吧。写好代码。搞定它。",
        "帮我跑一下这个分析。生成报告。处理这个数据。",
        "写一个函数。实现这个功能。建一个新的notebook。",
        "成立专家组，讨论一个可行计划并执行。",
        "专家组讨论并按计划执行。继续下一步。直接跑完。",
        "帮我配一下环境。安装这个包。设置好数据库。",
        "把这些整理成正式的交付。部署到生产环境。",
        "写一下这个部分。帮我总结一下。整理一下格式。",
        "加一个搜索功能。添加错误处理。做成响应式的。",
        "直接进行下一步的任务吧。开始吧。继续。",
    ],
    "hypothesis_1_zh": [
        # Symptom only — something's wrong but no specifics
        "有个报错。代码跑不通。结果不对。",
        "模型效果不好。数据有问题。输出不对。",
        "这个不work。测试没通过。有什么地方不对。",
        "构建失败了。安装不上。运行不了。",
        "效果不太理想。数字对不上。有些奇怪。",
        "strict太低了。coverage不够。结果不达标。",
        "图表看起来不对。分析结果有问题。",
        "notebook打不开了。文件有问题。",
    ],
    "hypothesis_2_zh": [
        # Locates the problem — identifies where/when it fails
        "这个不对吧，再检查一下，最后冻结的应该是459。",
        "0.9412 / 0.3500 / 0.3500是pilot侧的吗？",
        "endpoint pool本身大约只占split的19.6%，这是什么意思？",
        "runway weight的endpoint pool比例和main还有boundary差很多？",
        "pilot模型不是只看两端吗？为什么coverage能到75%？",
        "那全split的话占比多少？后续coverage是否应该优先看整体？",
        "错误出现在第42行。在提交空表单的时候会失败。",
        "只有在缺失值的时候才会报错。小数据没问题，大数据就崩溃。",
        "Q3的数据有异常，但Q1没有。",
        "训练集上表现好但测试集上不行。",
    ],
    "hypothesis_3_zh": [
        # Forms a hypothesis — proposes a cause
        "我觉得问题是session没有在middleware之前初始化。",
        "我怀疑模型过拟合了——训练准确率比验证高太多。",
        "但是并不是任务目标变了才导致结果变好吧？",
        "冻结基线对的口径是什么？原来的路径上没有优化空间了吗？",
        "我觉得是因为特征共线性导致feature importance有误导性。",
        "我猜是DNS没有解析因为容器网络是隔离的。",
        "我觉得根本原因是X，因为Y只在Z为真的时候发生。",
        "我怀疑是环境变量没有加载因为.env文件不在构建上下文里。",
        "可能是缓存的问题。我觉得是之前的结果没有清掉。",
        "我的假设是A是由B引起的，不是C，因为模式匹配B。",
    ],
    "hypothesis_4_zh": [
        # Tested a hypothesis — tried something, drew conclusion
        "我试过换模型了，tree和lgb都试了。",
        "我已经把端口改成8080了，能跑，所以是别的东西占了80。",
        "试了一下去掉异常值重新跑，相关性减弱了，确认是它们导致的。",
        "我已经排除了X和Y，分别测试过了，唯一的解释就是Z。",
        "我测了小数据集，模式一样，所以不是样本量的问题。",
        "标准化之后问题依然存在，所以可能是模型架构的问题。",
        "两种方案都试了——A更快但B更准。考虑到我们的限制，B更好。",
        "用新的容器测试了可以跑，所以问题在缓存的镜像层。",
        "前面做的那些探索都失败了，直到我们用blast分train val之后才改善。",
        "我试了重构论点但同样的反驳依然成立。也许前提本身就是错的。",
    ],

    # Ownership
    "user_driven_zh": [
        "我想理解这个是怎么工作的。能解释一下取舍吗？",
        "我设计的架构是这样的，你觉得合理吗？",
        "我想用X方案。你觉得需要改什么？",
        "我的想法是先做认证，再搞UI。",
        "不，我不想那样做。我想保持简单。",
        "我觉得咱们要规范化一下。",
        "用另一个方案吧。我觉得这样更好。",
        "先不压迫修改别的notebook了。",
        "pilot侧strict要尽可能的高，coverage可以先不管。",
        "我希望提交的这版可以直接运行得到结果。",
        "不，要把所有的数据处理流程全部写进这个notebook。",
        "我想问的就是相比于原来的探索我们做了什么改善。",
    ],
    "ai_driven_zh": [
        "你觉得该怎么做？什么方案最好？",
        "你决定吧。你比较懂。随便你。",
        "你推荐什么？听你的。你说了算。",
        "哪个好？我不太确定。都行吧。",
        "我没有偏好。两个都可以。你选。",
        "我应该先做什么？重点在哪？",
        "不知道用哪个框架好。你建议呢？",
        "该怎么组织？什么方式最合适？",
        "好的按你说的来。嗯行。",
    ],

    # Critical engagement
    "critical_zh": [
        "等一下，这好像不对。我觉得应该不是这样。",
        "你确定吗？我记得不是这样运行的。",
        "这个方案如果遇到X情况就会出问题。你漏了边界情况。",
        "我检查了你的方案，有bug。让我解释一下。",
        "这个不对吧。文档上说的不一样。",
        "你说的和我看到的结果不一致。有问题。",
        "前面我们也做了很多尝试但都没成功。",
        "但是并不是任务目标变了才导致结果变好吧？",
        "我说的不是这个意思。再看一遍。",
        "这个数据有点夸张了。再确认一下。",
        "我跑了你的代码结果不一样。哪里有问题。",
    ],
    "passive_zh": [
        "好的。没问题。可以。行。",
        "看起来不错。就这样吧。",
        "谢谢，很好。完美。就用这个。",
        "好的我直接用了。好。明白了。",
        "可以先这样吧。就这么办。继续。",
        "嗯好。收到。了解。",
        "有道理。就这样。",
        "OK。下一个。没问题。",
    ],

    # Self-reliance
    "self_reliant_zh": [
        "我试过X但是不行。我已经debug了一个小时了。",
        "我自己想出来了但想确认一下。有没有更好的方法？",
        "我查了文档但没有覆盖这个情况。",
        "我搜了Stack Overflow试了排名靠前的答案，都不行。",
        "之前我们也做了很多尝试。前面做了那些探索都失败了。",
        "我缩小范围到这三行了。我觉得我知道问题在哪。",
        "基本弄好了，就差最后一部分。",
        "我的方案对A有效但对B无效。",
        "我试过两种方案了。A更快但B更准。",
        "已经试过常规方法了，没用。",
    ],
    "not_self_reliant_zh": [
        "完全不知道从哪开始。我不会做这个。",
        "我放弃了。帮我修吧。我看不懂这个报错。",
        "太难了。帮我做吧。不想想了。",
        "怎么做？完全没头绪。",
        "能不能直接帮我搞定？不懂。",
        "这个我不理解。解释不了。",
        "不知道搜什么。完全没方向。",
        "从来没做过这个。能不能手把手教？",
    ],

    # Metacognition
    "metacognitive_zh": [
        "我的思路对吗？是不是方向搞错了？",
        "我对这个的理解可能不对。能帮我审查一下我的思路吗？",
        "我可能有盲点。这样思考对不对？",
        "我想确保我理解了，不是单纯复制答案。",
        "我是不是把这个搞复杂了？",
        "可能我从根本上就想错了。",
        "先别给答案，能帮我理解为什么会这样吗？",
        "我想学会方法，不只是要结果。",
        "我觉得我对基本概念可能还不够理解。",
        "我的心智模型可能有问题。帮我想想。",
    ],
    "not_metacognitive_zh": [
        "直接给答案。不需要解释。",
        "别解释了，给我看代码就行。",
        "太长了，直接告诉我怎么做。",
        "不管为什么，修好就行。",
        "跳过理论，给我方案。",
        "简单说就行。TLDR。",
        "直接告诉我输入什么命令。",
    ],

    # Delegation
    "delegation_zh": [
        "直接做吧。帮我写。全部搞定。",
        "你来处理吧。你觉得最好怎么做就怎么做。",
        "实现这个功能。写这个函数。生成代码。",
        "帮我写一下。帮我处理。请实现。",
        "做完剩下的。继续做。搞定它。",
        "专家组讨论并执行。按计划执行。直接跑完。",
        "你来想方案然后直接做。",
        "全部帮我搞好。不想管了。",
        "整理成正式的交付。打包好。",
        # "帮我 + verb" pattern — key ZH delegation signal
        "帮我修一下这个。帮我做这个。帮我搞定。帮我弄好。",
        "帮我修这个bug。帮我改一下。帮我处理这个问题。",
        # Helplessness + delegation
        "完全不知道怎么办，帮我弄。不知道从哪开始，帮我做吧。",
        "搞不定，你来吧。放弃了，帮我处理。我不会，帮我做。",
        "完全不知道从哪开始，你来做吧。不知道怎么下手，帮我搞定。",
        "不知道怎么弄，你来处理吧。没有头绪，帮我做吧。",
    ],
    "not_delegation_zh": [
        "我试了这个方案想知道为什么不行。",
        "这是我的尝试，哪里做错了？",
        "能解释一下原理吗？我自己来修。",
        "我想先了解取舍再做决定。",
        "给我提示就行，别直接给答案。",
        "帮我审查一下思路，别直接重写。",
        "我想学会方法，不只是要结果。",
        "需要理解什么概念才能修好这个？",
        # "自己来" pattern — self-implementation intent
        "我自己来写代码，给我提示就行。",
        "给我方向就好，代码我自己写。我自己来做，只要告诉我思路。",
        "我自己来实现，你帮我理清逻辑就行。",
    ],

    # Task domains
    "task_code_zh": [
        "写一个函数。修这个bug。实现这个功能。",
        "代码报错了。debug一下。跑不通。",
        "重构这段代码。加单元测试。代码review。",
        "怎么在React里实现X？帮我写SQL查询。",
        "这个类有问题。API调用失败了。测试没通过。",
    ],
    "task_data_zh": [
        "分析这个数据集。训练一个分类器。建模预测。",
        "跑个统计检验。做个可视化。清洗数据。",
        "模型效果不好。应该用什么特征？过拟合了。",
        "画个图。算相关性。评估模型准确率。",
        "特征工程。数据预处理。交叉验证。coverage。strict。",
    ],
    "task_writing_zh": [
        "写一个引言。编辑这段话。",
        "总结这篇文章。帮我写邮件。",
        "改简洁一点。改通顺一点。改专业一点。",
        "写封求职信。生成报告。润色一下。",
    ],
    "task_research_zh": [
        "解释一下transformer怎么工作的。X是什么？",
        "X和Y有什么区别？底层原理是什么？",
        "总结这篇论文。X方案的优缺点是什么？",
        "我想更深入了解X。教我Y。",
    ],
    "task_planning_zh": [
        "帮我设计架构。怎么组织项目结构？",
        "该用X还是Y？什么设计模式好？",
        "帮我做个计划。分析一下风险。下一步怎么做？",
        "权衡一下利弊。帮我在这几个方案里选。",
    ],
    "task_config_zh": [
        "怎么安装X？配一下环境。设置好工具。",
        "Docker跑不起来。部署到生产。配置CI/CD。",
        "包装不上。构建失败。权限不够。端口被占了。",
        "配置环境变量。写Dockerfile。设置数据库。",
    ],
}

# ── Per-type prototype overrides ─────────────────────────────────────────────
# These capture how signals manifest *differently* in non-coding sessions.
# At classification time, these are blended with the base prototypes when
# the session type is known, giving a more accurate read.
#
# Key insight: "explain X to me" is hypothesis_0 in a coding context (lazy)
# but legitimate engagement in a research context.

SESSION_TYPE_OVERRIDES: dict[str, dict[str, list[str]]] = {
    "research": {
        # In research, asking deep questions IS engagement — not delegation
        "hypothesis_0": [
            # Only truly zero-effort research asks
            "Tell me about X. What is X. Give me info on X.",
            "Just summarize it. Give me the answer.",
            "告诉我X是什么。简单说一下。给我答案。",
        ],
        "hypothesis_2": [
            # Focused, specific research questions = locating the problem space
            "What's the difference between X and Y in terms of Z?",
            "How does X handle the case where Y happens?",
            "I read that X does Y — but how does that work with Z?",
            "Explain specifically how X interacts with Y under condition Z.",
            "X和Y在Z方面有什么区别？",
            "X在Y情况下是怎么处理的？",
            "我看到说X会导致Y——但这和Z怎么兼容？",
        ],
        "hypothesis_3": [
            # Research hypotheses — forming mental models
            "I think X works this way because of Y — is that right?",
            "My understanding is that X and Y are related because Z.",
            "I think the key difference is Z — the other factors don't matter as much.",
            "Based on what I've read, I believe X because Y.",
            "So if I understand correctly, X happens because of Y?",
            "我觉得X是这样运作的因为Y——对吗？",
            "我的理解是X和Y相关因为Z。",
            "基于我读到的，我认为X是因为Y。",
        ],
        # Research delegation is narrower — only "just give me answers" counts
        "delegation": [
            "Just give me the summary. Don't explain, just list the facts.",
            "I don't want to think about it. Just tell me the answer.",
            "直接给我总结。别解释了。直接告诉我答案。",
        ],
        "not_delegation": [
            "Can you explain how X works? I want to understand the concept.",
            "Walk me through the reasoning. Help me understand why.",
            "What's the intuition behind X? Teach me the fundamentals.",
            "I want to build a mental model of how X works.",
            "What's the difference between X and Y? How do they compare?",
            "How does X work under the hood? What are the tradeoffs?",
            "Why does X happen? What causes this? Can you teach me?",
            # Concrete conceptual questions (not just X/Y templates)
            "What's the difference between TCP and UDP? How do they compare?",
            "How does garbage collection work in different languages?",
            "What are the tradeoffs between SQL and NoSQL databases?",
            "How does encryption work? What makes it secure?",
            "Explain the difference between threads and processes.",
            "解释一下X怎么工作的？我想理解这个概念。",
            "帮我理解一下为什么。背后的逻辑是什么？",
            "X的直觉是什么？教我基础知识。",
            "X和Y有什么区别？怎么对比？为什么会这样？",
        ],
        # Ownership in research = steering the inquiry
        "user_driven": [
            "I specifically want to understand X, not Y.",
            "Let's focus on the Z aspect — that's what I need to grasp.",
            "I already know X, so skip that. I need to understand Y.",
            "Can we go deeper on this specific point?",
            "我想了解的是X，不是Y。",
            "我们重点看Z这个方面。",
            "X我已经知道了，我需要理解的是Y。",
        ],
    },
    "creative": {
        # In creative work, directing the AI IS ownership
        "hypothesis_0": [
            # Creative delegation is different — only totally passive counts
            "Write something. Just make something up. Whatever you want.",
            "写点什么。随便写。你随意。",
        ],
        "hypothesis_2": [
            # Specific creative direction = showing engagement
            "The tone should be melancholic but hopeful at the end.",
            "I want the opening to hook the reader with a question.",
            "Make it sound like a conversation between two old friends.",
            "Use shorter sentences for tension, longer ones for reflection.",
            "语气要忧郁但结尾要有希望。",
            "开头用一个问题吸引读者。",
            "写得像两个老朋友之间的对话。",
        ],
        "hypothesis_3": [
            # Creative hypotheses — trying an approach, knowing why
            "I think the piece needs more conflict in the middle section because the tension drops.",
            "The problem is the voice shifts between paragraphs — I want consistent first person.",
            "I think a metaphor about water would work better here because the theme is about flow.",
            "我觉得中间部分需要更多冲突因为张力下降了。",
            "问题是段落之间的声音在变——我想要一致的第一人称。",
            "我觉得用水的比喻更好因为主题是关于流动的。",
        ],
        # Creative delegation is much narrower
        "delegation": [
            "Write the whole thing, I don't care how. Just finish it.",
            "Whatever style you want. I have no preferences.",
            "全部你来写。我不管。随便什么风格。",
        ],
        "not_delegation": [
            "Write a poem about autumn with a bittersweet tone.",
            "Draft an email that's professional but warm.",
            "Help me write an introduction — I want it to start with an anecdote.",
            "Make it more concise but keep the emotional impact.",
            "写一首关于秋天的诗，带点苦涩。",
            "帮我写一封专业但温暖的邮件。",
            "帮我写个开头——我想用一个小故事开始。",
        ],
        # Creative ownership = aesthetic direction
        "user_driven": [
            "Make it darker. I want a more ironic tone.",
            "No, that's too formal. Make it conversational.",
            "I like the structure but the ending needs to be stronger.",
            "Change the metaphor — use fire instead of water.",
            "太正式了。改口语化一点。",
            "结构可以但结尾要更有力。",
            "换个比喻——用火不用水。",
        ],
        # Creative critical engagement = iterating on drafts
        "critical": [
            "That doesn't capture the mood I wanted. It should feel more urgent.",
            "No, that's too cheerful. Make it darker. The tone is wrong.",
            "The second paragraph loses the reader. Too many details.",
            "Good start but the ending is weak. Needs a stronger closing line.",
            "This sounds generic. Make it more specific to my situation.",
            "That's not what I asked for. Try again with a different angle.",
            "The style is off. I wanted something more poetic, less prosaic.",
            "没有抓到我想要的感觉。应该更紧迫。",
            "不对，太欢快了。改暗一点。语气不对。",
            "第二段太啰嗦了。细节太多。",
            "结尾太弱了。需要更有力的收尾。",
        ],
    },
    "casual": {
        # Casual chat has relaxed expectations — short messages are normal
        "delegation": [
            # Only aggressive outsourcing counts as delegation in casual
            "Figure everything out for me. Handle my whole day.",
            "帮我全部搞定。别让我想了。",
        ],
        "not_delegation": [
            "What do you think about X? Quick question.",
            "Any recommendations? What's your take?",
            "你觉得X怎么样？快速问一下。推荐一下。",
        ],
    },
}

_model = None
_prototype_embeddings: dict[str, np.ndarray] = {}


def _get_model():
    """Lazy-load the embedding model (first call only)."""
    global _model
    if _model is None:
        try:
            import logging
            import warnings
            from sentence_transformers import SentenceTransformer

            # Suppress noisy HF/transformers warnings that confuse first-time users
            import os
            os.environ.setdefault("HF_HUB_VERBOSITY", "error")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", module="huggingface_hub.*")

            _MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

            # paraphrase-multilingual-MiniLM-L12-v2: 50+ languages, ~118MB
            # Maps cross-lingual meaning to same embedding space —
            # English prototypes correctly classify Chinese/French/etc. input.
            _model = SentenceTransformer(_MODEL_NAME)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
    return _model


def _get_prototype_embeddings() -> dict[str, np.ndarray]:
    """Compute prototype embeddings once, cache them.

    Builds separate embeddings for English (PROTOTYPES) and Chinese (PROTOTYPES_ZH)
    so that each language keeps its own focused centroid — no dilution.
    Classification uses max(en, zh) similarity.

    Also merges any user-learned prototypes saved by Tier 3 in
    ~/.mindcheck/learned_prototypes.json.
    """
    global _prototype_embeddings
    if not _prototype_embeddings:
        model = _get_model()

        # Start from a copy of built-in prototypes (English)
        extended: dict[str, list[str]] = {k: list(v) for k, v in PROTOTYPES.items()}

        # Merge learned prototypes saved by Tier 3
        # Supports both hypothesis levels AND boolean signals (v1.2+)
        import json as _json
        from pathlib import Path
        learned_path = Path.home() / ".mindcheck" / "learned_prototypes.json"
        if learned_path.exists():
            try:
                learned = _json.loads(learned_path.read_text(encoding="utf-8"))
                for entry in learned:
                    text = entry.get("text", "").strip()
                    if not text:
                        continue

                    # Hypothesis level learning
                    level = entry.get("level")
                    if isinstance(level, int) and 0 <= level <= 4:
                        extended[f"hypothesis_{level}"].append(text)

                    # Boolean signal learning (v1.2+)
                    _SIGNAL_MAP = {
                        "is_user_driven":   ("user_driven",    "ai_driven"),
                        "is_critical":      ("critical",       "passive"),
                        "is_self_reliant":  ("self_reliant",   "not_self_reliant"),
                        "is_metacognitive": ("metacognitive",  "not_metacognitive"),
                        "is_delegation":    ("delegation",     "not_delegation"),
                    }
                    for signal_key, (pos_key, neg_key) in _SIGNAL_MAP.items():
                        val = entry.get(signal_key)
                        if val is True and pos_key in extended:
                            extended[pos_key].append(text)
                        elif val is False and neg_key in extended:
                            extended[neg_key].append(text)
            except Exception:
                pass  # Corrupt file — fall back to built-ins only

        # Encode English prototypes
        for key, texts in extended.items():
            combined = " ".join(texts)
            _prototype_embeddings[key] = model.encode(combined, normalize_embeddings=True)

        # Encode Chinese prototypes (separate centroids)
        for key, texts in PROTOTYPES_ZH.items():
            combined = " ".join(texts)
            _prototype_embeddings[key] = model.encode(combined, normalize_embeddings=True)

    return _prototype_embeddings


# ── Override prototype embeddings (per session type) ─────────────────────────
_override_embeddings: dict[str, dict[str, np.ndarray]] = {}


def _get_override_embeddings(session_type: str) -> dict[str, np.ndarray]:
    """Compute and cache override prototype embeddings for a session type."""
    global _override_embeddings
    if session_type not in _override_embeddings:
        overrides = SESSION_TYPE_OVERRIDES.get(session_type, {})
        if not overrides:
            _override_embeddings[session_type] = {}
            return _override_embeddings[session_type]

        model = _get_model()
        embs = {}
        for key, texts in overrides.items():
            combined = " ".join(texts)
            embs[key] = model.encode(combined, normalize_embeddings=True)
        _override_embeddings[session_type] = embs

    return _override_embeddings[session_type]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both already normalized


def _classify_message(text: str, session_type: str = "coding") -> dict:
    """
    Classify a single user message against all prototype categories.
    Returns similarity scores (0–1) for each signal.

    When session_type is not "coding", type-specific override prototypes are
    blended in: for each signal that has an override, the override similarity
    is averaged with the base similarity, nudging the classification towards
    type-appropriate behavior.
    """
    model = _get_model()
    protos = _get_prototype_embeddings()
    overrides = _get_override_embeddings(session_type) if session_type != "coding" else {}

    msg_emb = model.encode(text, normalize_embeddings=True)

    def sim(key: str) -> float:
        return _cosine_sim(msg_emb, protos[key])

    def sim_bilingual(key: str) -> float:
        """Max of English and Chinese prototype similarity (no dilution).
        If an override exists for this key, blend it in to shift the centroid
        towards session-type-appropriate patterns.
        Delegation/not_delegation use 70/30 blend (stronger override) because
        these signals differ most across session types."""
        en = sim(key)
        zh_key = f"{key}_zh"
        zh = sim(zh_key) if zh_key in protos else en
        base = max(en, zh)

        if key in overrides:
            override_score = _cosine_sim(msg_emb, overrides[key])
            # Stronger blend for delegation signals (differ most across types)
            if "delegation" in key:
                return override_score * 0.7 + base * 0.3
            return override_score * 0.6 + base * 0.4

        return base

    # Hypothesis level (0–4 by highest similarity across both languages)
    hyp_scores = {}
    for level in range(5):
        base_key = f"hypothesis_{level}"
        en = sim(base_key)
        zh_key = f"hypothesis_{level}_zh"
        zh = sim(zh_key) if zh_key in protos else 0.0
        base = max(en, zh)

        # Blend override if available
        if base_key in overrides:
            override_score = _cosine_sim(msg_emb, overrides[base_key])
            hyp_scores[level] = override_score * 0.6 + base * 0.4
        else:
            hyp_scores[level] = base

    sorted_hyp = sorted(hyp_scores.items(), key=lambda x: x[1], reverse=True)
    hypothesis_level = sorted_hyp[0][0]
    # Confidence = gap between top and runner-up. Small gap = uncertain classification.
    hypothesis_confidence = sorted_hyp[0][1] - sorted_hyp[1][1]

    # Task domain — highest similarity wins (bilingual, no overrides needed)
    _TASK_DOMAINS = ("task_code", "task_data", "task_writing",
                     "task_research", "task_planning", "task_config")
    task_domain_sim = lambda key: max(
        sim(key), sim(f"{key}_zh") if f"{key}_zh" in protos else 0.0
    )
    task_domain = max(_TASK_DOMAINS, key=task_domain_sim).replace("task_", "")

    return {
        "hypothesis_level":      hypothesis_level,
        "hypothesis_confidence": hypothesis_confidence,   # 0.0 = tie, ~0.1+ = confident
        "task_domain":           task_domain,
        "is_user_driven":    sim_bilingual("user_driven")    > sim_bilingual("ai_driven"),
        "is_critical":       sim_bilingual("critical")       > sim_bilingual("passive")          + 0.05,
        "is_self_reliant":   sim_bilingual("self_reliant")   > sim_bilingual("not_self_reliant") + 0.05,
        "is_metacognitive":  sim_bilingual("metacognitive")  > sim_bilingual("not_metacognitive"),
        "is_delegation":     sim_bilingual("delegation")     > sim_bilingual("not_delegation")   + 0.02,
        "low_confidence":    hypothesis_confidence < 0.04,  # flag for prototype review
    }


def extract_semantic(session: Session, session_type: str = "coding") -> SemanticSignals:
    """Run Tier 2 embedding classification on all user messages.

    When session_type is provided, per-type prototype overrides are used
    to adjust classification for non-coding conversations.
    """
    sig = SemanticSignals()
    user_msgs = session.user_messages

    if not user_msgs:
        return sig

    classifications = []
    for msg in user_msgs:
        text = msg.content.strip()
        # Skip messages too short to classify meaningfully —
        # single words / button clicks / confirmations add noise, not signal.
        if len(text) < 12:
            continue
        result = _classify_message(text, session_type=session_type)
        result["content"] = text[:100]
        classifications.append(result)

    if not classifications:
        return sig

    n = len(classifications)
    sig.hypothesis_level_avg = sum(c["hypothesis_level"] for c in classifications) / n
    sig.ownership_score = sum(1 for c in classifications if c["is_user_driven"]) / n
    sig.critical_engagement = sum(1 for c in classifications if c["is_critical"]) / n
    sig.self_reliance = sum(1 for c in classifications if c["is_self_reliant"]) / n
    sig.metacognition_score = sum(1 for c in classifications if c["is_metacognitive"]) / n
    sig.delegation_penalty = sum(1 for c in classifications if c["is_delegation"]) / n
    sig.per_message = classifications

    # ── Task domain breakdown ─────────────────────────────────────────────────
    raw_breakdown: dict[str, dict] = {}
    for c in classifications:
        domain = c["task_domain"]
        if domain not in raw_breakdown:
            raw_breakdown[domain] = {"count": 0, "hyp_sum": 0, "deleg_count": 0}
        raw_breakdown[domain]["count"] += 1
        raw_breakdown[domain]["hyp_sum"] += c["hypothesis_level"]
        raw_breakdown[domain]["deleg_count"] += 1 if c["is_delegation"] else 0

    sig.task_breakdown = {
        domain: {
            "count":           d["count"],
            "hypothesis_avg":  round(d["hyp_sum"] / d["count"], 2),
            "delegation_rate": round(d["deleg_count"] / d["count"], 2),
        }
        for domain, d in raw_breakdown.items()
        if d["count"] >= 2   # only report domains with at least 2 messages
    }

    return sig
