# BrowseComp evaluation prompts
# Based on the simple-evals BrowseComp prompt template

# Runtime prompt templates used by run.py
BROWSECOMP_RUNTIME_WITH_GET_DOCUMENT = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search and get_document tools provided. Please perform reasoning and use the tools step by step, in an interleaved manner. You may use the search and get_document tools multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

BROWSECOMP_RUNTIME_SEARCH_ONLY = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

BROWSECOMP_RUNTIME_SEARCH_ONLY_NO_CITATION = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

BROWSECOMP_BEST_TOOL_FIRST_PROMPT = """
You are a deep research agent with access to multiple search tools.

Question: {Question}

Goal: get the most accurate final answer.

Tool policy:
- Use tools actively when external information is needed.
- Choose the tool that is most likely to return the best evidence for the current information gap.
- Reuse the same tool when it is working well.
- Switch tools only when results are weak, conflicting, stale, or missing key facts.
- You may call tools multiple times until you have enough evidence.

When you have enough evidence, stop calling tools and give your final answer.

Your final response should be in the following format:
Explanation: {{your explanation for your final answer. Cite supporting docids inline in square brackets, e.g. [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

# Final answer prompt when max turns reached
BROWSECOMP_FINAL_ANSWER_PROMPT = """Based on all the information gathered so far, please provide your final answer to the question. You have reached the maximum number of tool calls, so you must now answer based on what you've learned.

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

BROWSECOMP_AGENT_PROMPTS = {
    # Runtime defaults for BrowseCompPlus generation
    "runtime_with_get_document": BROWSECOMP_RUNTIME_WITH_GET_DOCUMENT,
    "runtime_search_only": BROWSECOMP_RUNTIME_SEARCH_ONLY,
    "runtime_search_only_no_citation": BROWSECOMP_RUNTIME_SEARCH_ONLY_NO_CITATION,
    "best_tool_first": BROWSECOMP_BEST_TOOL_FIRST_PROMPT,
}

def resolve_prompt_key(
    prompt_key: str | None,
    include_get_document: bool,
) -> str:
    """Resolve runtime prompt selection."""
    if prompt_key:
        if prompt_key not in BROWSECOMP_AGENT_PROMPTS:
            raise ValueError(f"Unknown prompt key: {prompt_key}")
        return prompt_key

    if include_get_document:
        return "runtime_with_get_document"
    return "runtime_search_only"

def build_agent_messages(question: str, prompt_key: str, max_turns: int | None = None) -> list[dict]:
    """Build chat messages for BrowseCompPlus prompts."""
    if prompt_key not in BROWSECOMP_AGENT_PROMPTS:
        raise ValueError(f"Unknown prompt key: {prompt_key}")

    template = BROWSECOMP_AGENT_PROMPTS[prompt_key]
    has_inline_question = "{Question}" in template

    format_kwargs = {"Question": question}
    if "{max_turns}" in template:
        if max_turns is None:
            raise ValueError(f"Prompt '{prompt_key}' requires max_turns")
        format_kwargs["max_turns"] = max_turns

    try:
        rendered = template.format(**format_kwargs)
    except KeyError as exc:
        missing = str(exc)
        raise ValueError(f"Prompt '{prompt_key}' is missing placeholder value: {missing}") from exc

    if has_inline_question:
        return [{"role": "user", "content": rendered}]
    return [
        {"role": "system", "content": rendered},
        {"role": "user", "content": question},
    ]

BROWSECOMP_TOOL_DESCRIPTION_PROMPTS = {
    "detailed": {
        "pre": """Analyze search tool performance across N question-answering trajectories to generate improved tool descriptions that clearly differentiate when to use each tool.

<input>
- Question-answering trajectories with tool calls and results
- Search results returned by each tool (content may vary by tool)
- Whether the final answer was correct or incorrect
- Current tool descriptions
</input>

<analysis_requirements>
For each tool:
- Identify consistent patterns in the type and quality of results it returns
- Determine what distinguishes it from other tools
- Provide concrete proof: cite specific trajectories and queries showing these patterns
- Focus on situations where this tool performs differently than others

Evaluation notes:
- IMPORTANT: Compare tools relatively, not absolutely
- A tool is effective if it helps the agent reach the correct answer
- Consider both successful and unsuccessful trajectories
- Focus on: result relevance, information completeness, and query type suitability
- Don't dismiss a tool just because it was used in failed trajectories - focus on whether it provided useful information compared to alternatives
</analysis_requirements>

<output_per_tool>
**Tool: [name]**

Observed patterns: [Key behaviors identified with specific trajectory evidence]

Distinguishing characteristics: [What makes this tool different from others, with examples]

Updated description:
[Concise description stating when to use this tool relative to others]

Reasoning: [Justification based on trajectory evidence]
</output_per_tool>

<final_output>
After analyzing all tools, provide a decision framework for selecting between tools based on the patterns discovered.
</final_output>

Key: Every claim must reference trajectories. Descriptions must be comparative (tool X better than Y for Z) not absolute."""
    },
    "detailed_v2": {
        "pre": """Analyze search tool performance across N question-answering trajectories to discover what distinguishes each tool and write descriptions that differentiate when to use each.

<discovery_task>
**Goal: Discover what makes each tool behaviorally different.**

Tools may differ by domain coverage, result quality, recency, query specialization, or other patterns. Don't assume - discover through evidence.
</discovery_task>

<input>
- Trajectories with tool calls and results (URLs, titles, snippets)
- Answer correctness
- Current tool descriptions
</input>

<analysis_approach>
For each tool, examine:

**Observable Signals:**
- URLs: What domains appear? Consistent patterns?
- Content style: Encyclopedic, academic, news-like, technical?
- Recency: Fresh vs historical content?
- Query patterns: What question types succeed vs fail?

**Comparative Differences:**
- For similar queries, how do results differ between tools?
- What does THIS tool do better/worse than others?
- When is this tool the right choice vs alternatives?

**Evidence Requirements:**
- Cite specific trajectories and examples for every claim
- Quantify when possible: "7 of 10 academic queries succeeded"
- Note both supporting and contradicting evidence
- Distinguish strong patterns (consistent, many examples) from weak ones (noisy, few examples)

**Critical:**
- Compare relatively (tool X vs Y for Z), not absolutely
- A tool is effective if results help reach correct answers
- Don't dismiss tools used in failed trajectories - evaluate if they provided relevant information
</analysis_approach>

<output_format>
**CRITICAL: You MUST output each tool in EXACTLY this format. Do not add preambles, summaries, or skip any sections.**

For each tool, provide:

Tool: [tool_name]

Distinguishing characteristics:
[What behavioral patterns differentiate this tool? 2-3 sentences]

Evidence:
[Specific trajectories, queries, and URLs supporting these patterns. Use concrete examples.]

Comparative strengths/weaknesses:
[When is this tool better/worse than alternatives? Give specific examples.]

Updated description:
[REQUIRED: One clear, concise sentence describing when to use this tool]

Reasoning:
[Brief justification based on trajectory evidence]

---

**Repeat this exact format for EVERY tool. Do not add commentary between tools or at the end.**
</output_format>

<final_output>
After analyzing all tools individually, provide:

**Tool Selection Framework:**
- Key differentiators between tools
- Selection heuristics: given query characteristics, which tools are promising?
- Confidence notes: which distinctions are clear vs uncertain
</final_output>

Key: Ground every claim in trajectory evidence. Focus on strong, consistent patterns."""
    }
}

BROWSECOMP_SYNTHESIS_DESCRIPTION_PROMPTS = {
   "v1": """You will receive N LLM responses, each analyzing different batches of BrowseComp question-answering trajectories. Synthesize these into definitive tool descriptions.

<synthesis_task>
For each tool:
1. Identify patterns that appear across multiple responses
2. Note contradictions between responses
3. Distinguish true patterns from batch-specific noise
4. Look for emergent patterns that no single analysis identified but become visible when viewing all analyses together
5. Create ONE final description based on the most reliable patterns

Critical:
- A behavior mentioned in only 1-2 responses is likely batch-specific noise
- Focus on patterns that multiple independent analyses discovered
- Also identify meta-patterns: behaviors that emerge from the collective evidence but weren't explicitly stated in any single response
- When responses conflict, examine their evidence strength
- Final descriptions should capture the tool's strengths/weaknesses but NOT explicitly name other tools
</synthesis_task>

<output_format>
**Tool: [name]**

Synthesis reasoning:
[Explain which patterns were most consistent across analyses, what emergent patterns were discovered, how conflicts were resolved, and why certain behaviors were included/excluded in the final description.]

Final description:
[Single definitive description of when to use this tool. Describe its characteristics and optimal use cases WITHOUT referencing other tools by name. Example: "Best for queries requiring recent information or real-time data. Returns comprehensive results with detailed snippets. May be less effective for historical or archival content."]
</output_format>"""
,
   "v2": """You will receive N LLM responses, each analyzing different batches of BrowseComp question-answering trajectories. Synthesize them into one reliable, production-quality description per tool.

<synthesis_task>
For each tool:
1. Collect candidate patterns from all batch analyses.
2. Weight each pattern by support strength:
   - High confidence: repeated across many batches with consistent evidence.
   - Medium confidence: appears in multiple batches with minor conflict.
   - Low confidence: appears rarely or conflicts substantially.
3. Resolve conflicts explicitly:
   - Prefer broader, repeatedly supported patterns.
   - Keep narrow claims only if evidence is strong and specific.
4. Produce one final operational description the runtime agent can follow.

Critical rules:
- Do not invent frequencies, counts, or statistics.
- If evidence is weak/contradictory, write a conservative description and say so in reasoning.
- Do not mention other tools by name in final descriptions.
- Prioritize actionable behavior over stylistic prose.
</synthesis_task>

<output_format>
Use exactly this structure for each tool:

**Tool: [name]**

Synthesis reasoning:
[2-5 sentences: which behaviors were strongly supported, which were uncertain, and how conflicts were resolved.]

Final description:
[1-2 short sentences, action-oriented. Prefer this template:
"Use when ... Avoid when ..."
If "Avoid when" is not justified by evidence, omit it.]
</output_format>

<quality_bar>
- Final descriptions must be concise and directly usable for tool selection at inference time.
- Keep only high-signal differentiators; remove batch-specific noise.
</quality_bar>"""
}

# Lightweight prompts used by generate_improved_descriptions.py

