# Chess agent prompts with different playing styles
CHESS_AGENT_PROMPTS = {
   "optimized_single": """You are a chess-playing AI assistant with access to chess analysis tools for evaluating positions and suggesting moves.

<objective>
Play the strongest move in the current position by making exactly ONE tool call to analyze the position.
</objective>

<tool_selection_strategy>
- Make exactly ONE tool call per move decision
- Choose the strongest tool available for the current position
- If tools appear similar in capability or are similarly unknown, consider trying a different tool than recently used ones to diversify your information sources
- When one tool is clearly superior for the position type, use it regardless of history
</tool_selection_strategy>

<exploration_principle>
- Primary goal: Select the strongest tool for each position
- Secondary consideration: If multiple tools seem equally strong or equally unknown, vary your selection based on recent usage history
- This diversification helps avoid potential blind spots from relying on a single tool's perspective
- Never sacrifice move quality for exploration - only explore when tools are genuinely comparable
</exploration_principle>

<decision_framework>
With your single tool call, consider:
- What type of position is this? (tactical, positional, endgame, opening)
- Which tool is strongest for this specific position?
- If multiple tools seem equally strong, which have I used recently?
- Is there a clear best tool, or are several tools comparably suitable?
</decision_framework>

<tool_preamble>
Before making your tool call:
- Explain which tool you're selecting and why it's the strongest choice for this position
- If multiple tools seemed equally viable, briefly note why you selected this one over the others
</tool_preamble>

<quality_checks>
- Select the strongest available tool (or make a reasonable choice among equals)
- Make exactly one tool call
</quality_checks>""",

}

# Tool description improvement prompts for chess
CHESS_TOOL_DESCRIPTION_PROMPTS = {
    "detailed": {
        "pre": """Analyze chess tool performance across N game trajectories to generate improved tool descriptions that clearly differentiate when to use each tool.

<input>
- Game trajectories with tool calls, moves, and positions
- Board evaluations (positive=White advantage, negative=Black advantage)
- Current tool descriptions
- Side played by agent in each game
</input>

<analysis_requirements>
For each tool:
- Identify consistent patterns in its behavior and performance
- Determine what distinguishes it from other tools
- Provide concrete proof: cite specific trajectories and moves showing these patterns
- Focus on situations where this tool performs differently than others

Evaluation notes:
- Higher eval is better for White, lower eval is better for Black
- IMPORTANT: Always compare tools relatively, not absolutely
- Example for White: Tool A suggesting move to +2 is better than Tool B suggesting +1
- Example for Black: Tool A suggesting move to -3 is better than Tool B suggesting -1
- Critical: Even in losing positions, compare which tool finds the best continuation
  * For White: -5 is much better than -10 (both losing, but one is more resilient)
  * For Black: +10 is much better than +15 (both losing, but one offers more resistance)
- Don't dismiss a tool just because it suggested moves in bad positions - focus on whether it found the BEST move among the alternatives
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
    }
}

CHESS_SYNTHESIS_DESCRIPTION_PROMPTS = {
   "v1": """You will receive N LLM responses, each analyzing different batches of chess game trajectories. Synthesize these into definitive tool descriptions.

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
[Single definitive description of when to use this tool. Describe its characteristics and optimal use cases WITHOUT referencing other tools by name. Example: "Best for tactical positions requiring deep calculation. Excels at finding forcing sequences and material sacrifices. Tends to be overly aggressive in quiet positions."]
</output_format>"""
}