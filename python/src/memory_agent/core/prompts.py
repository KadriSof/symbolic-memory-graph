"""
Unified Prompt Templates for the Cogito Agent.

All prompts are centralized here for easy maintenance and versioning.
"""


class Prompts:
    """
    Unified prompt templates for the Cogito Agent.

    Each prompt is designed for a specific step in the Cogito pipeline.
    All prompts use f-strings for variable interpolation.
    """

    # Step 1: COMPREHEND
    @staticmethod
    def comprehend(query: str) -> str:
        """Unified comprehension prompt."""
        from .schemas import get_llm_format_instructions, ComprehensionSchema

        format_instructions = get_llm_format_instructions(ComprehensionSchema)

        return f"""
        You are a precise query analyzer for a cognitive agent with symbolic memory.

        **IMPORTANT: Your goal is to choose the SIMPLEST path that correctly handles the query.**
        - REACT is fast, cheap, and sufficient for most queries.
        - COGITO is powerful but expensive — use it ONLY when absolutely necessary.

        Original user query:
        ---
        {query}
        ---

        Analyze the query and return a JSON object with the following fields:

        {format_instructions}

        **DECISION RULES for modus_operandi:**

        ## USE "REACT" (Simple, fast, cheap) when:
        - Factual questions: "What is the capital of France?"
        - Direct calculations: "What is 2 + 2?"
        - Simple translations: "Say hello in Spanish"
        - Single-step tool calls: "Get the weather in Paris"
        - Database queries: "Return the number of rows in the students column"
        - Direct lookups: "Find the user with ID 123"
        - No relationships between multiple entities
        - No need for memory or context from previous conversations

        ## USE "COGITO" (Complex, powerful, expensive) ONLY when:
        - The query explicitly asks about RELATIONSHIPS between entities
        - The query requires building UNDERSTANDING, not just retrieval
        - The query is about CONNECTIONS: "How is X connected to Y?"
        - The query requires DETECTING CONTRADICTIONS
        - The query involves MULTI-STEP reasoning with memory
        - The query is EXPLICITLY about the agent's memory: "What do you know about X?"
        - The query involves BUILDING a mental model: "Tell me about X and Y"

        **Key principle: When in doubt, choose REACT.**

        Return ONLY valid JSON, no additional text.

        Example outputs:
        1. Simple query → REACT:
        {{
            "reconstructed_query": "What is the capital of Tunisia?",
            "user_intent": "ask",
            "modus_operandi": "REACT",
            "active_context": "question_answering",
            "entities": [
                {{"id": "tunisia", "label": "Tunisia", "type": "country"}}
            ],
            "relations": [],
            "confidence": 0.95
        }}

        2. Complex query → COGITO:
        {{
            "reconstructed_query": "What is the relationship between Geralt and Yennefer?",
            "user_intent": "ask",
            "modus_operandi": "COGITO",
            "active_context": "lore_building",
            "entities": [
                {{"id": "geralt", "label": "Geralt of Rivia", "type": "character"}},
                {{"id": "yennefer", "label": "Yennefer of Vengerberg", "type": "character"}}
            ],
            "relations": [
                {{"source": "geralt", "target": "yennefer", "label": "loves", "direction": "asymmetric"}}
            ],
            "confidence": 0.9
        }}
        """

    # Step 3: CONSOLIDATE
    @staticmethod
    def consolidate(context: str, new_primitives: str, query: str) -> str:
        """Consolidation prompt."""
        from .schemas import get_llm_format_instructions, ConsolidationSchema

        format_instructions = get_llm_format_instructions(ConsolidationSchema)

        return f"""
        You are a knowledge consolidation engine. Your task is to compare current knowledge with new information and propose minimal, precise updates to the knowledge graph.

        Current knowledge (from the graph):
        {context}

        New information (extracted from the user query):
        {new_primitives}

        User query:
        {query}

        Analyze and return a JSON object with the following fields:

        {format_instructions}

        **CRITICAL INSTRUCTIONS:**
        1. **new_nodes / new_edges**: ONLY include entities/relations that are TRULY new and not already present in the current knowledge.
        2. **modified_nodes / modified_edges**: ONLY include existing entities/relations that need their metadata, labels, or weights updated based on the new information.
        3. **gaps**: This MUST be a list of simple STRING questions representing missing information needed to fully answer the query. DO NOT output objects or dictionaries here.
           - [X] CORRECT: "gaps": ["What is Yennefer's age?", "Who is Ciri's mother?"]
           - [!] INCORRECT: "gaps": [{{"id": "yennefer", "type": "sorceress"}}]
        4. **conflicts**: List any direct contradictions between current knowledge and new information.
        5. Return ONLY valid JSON. Do not include markdown formatting or conversational text outside the JSON object.
        """

    # Step 4: REASON
    @staticmethod
    def reason(
        query: str, context: str, gaps: str, tool_results: str | None = None
    ) -> str:
        """Reasoning prompt."""
        # Clean up empty gaps display for better LLM comprehension
        gaps_display = "None" if gaps.strip() in ("[]", "") else gaps

        tool_section = (
            f"\nTool results:\n{tool_results}"
            if tool_results and tool_results.strip() not in ("[]", "")
            else "\nNo tool results needed or available."
        )

        return f"""
        You are a reasoning agent with access to a symbolic knowledge graph.

        User query: {query}

        Knowledge graph context:
        {context}

        Identified gaps that need to be filled:
        {gaps_display}
        {tool_section}

        Think step by step:
        1. What do I need to solve this?
        2. How does the graph context help?
        3. What gaps must I fill (or can I answer without them)?
        4. What is the final solution?

        Provide your reasoning and solution clearly.
        """

    # Step 6: RESPOND
    @staticmethod
    def synthesize_response(
        reasoning: str,
        solution: str,
        confidence: float,
        graph_context: str,
        max_context: int = 1000,
    ) -> str:
        """Response synthesis prompt."""
        truncated_context = (
            graph_context[:max_context]
            if len(graph_context) > max_context
            else graph_context
        )

        return f"""
        Synthesize a clear, natural response to the user.

        Reasoning trace: {reasoning}
        Solution: {solution}
        Confidence level: {confidence}

        Graph context (for reference):
        {truncated_context}

        Respond to the user with:
        1. The answer or solution
        2. Any relevant caveats or confidence notes
        3. Suggested next steps (if applicable)

        Keep the response concise, helpful, and friendly.
        Do not mention the graph or reasoning process unless directly relevant.
        """

    # ReAct Fallback
    @staticmethod
    def react_system_prompt(tools: list | None = None) -> str:
        """ReAct system prompt with or without tools."""
        has_tools = bool(tools) and len(tools) > 0

        if has_tools and tools is not None:
            tool_desc = "\n".join(f"  - {t.name}: {t.description}" for t in tools)
            tool_section = f"""
            You have access to the following tools:
            {tool_desc}

            When you need to use a tool, follow this structure:
            <THOUGHT> reason about the current situation </THOUGHT>
            <TOOL> the tool to execute </TOOL>
            <TOOL-INPUT> {{"key": "value"}} </TOOL-INPUT>
            <TOOL-OUTPUT> result from the tool </TOOL-OUTPUT>
            """
        else:
            tool_section = """
            You have NO tools available.
            You CANNOT use <TOOL> or <TOOL-INPUT> markers.
            You must respond directly with <FINAL-ANSWER> after thinking.
            If you lack information, state this honestly.
            NEVER invent or hallucinate information.
            """

        return f"""
        You are a ReAct reasoning agent.

        {tool_section}

        When you have enough information to answer the user, respond with:
        <THOUGHT> I now know the final answer </THOUGHT>
        <FINAL-ANSWER> the final response to the user </FINAL-ANSWER>

        ## RULES
        1. Keep thoughts concise and task-oriented
        2. Never invent tool observations
        3. Tool Input must always be valid JSON (if tools are available)
        4. If no tools are available, skip TOOL and TOOL-INPUT
        5. Provide direct, truthful answers

        ## EXAMPLE WITH TOOLS
        Question: What is the weather in Paris?
        <THOUGHT> I need to get the weather for Paris</THOUGHT>
        <TOOL>get_weather</TOOL>
        <TOOL-INPUT>{{"city": "Paris"}}</TOOL-INPUT>
        <TOOL-OUTPUT>Sunny, 25°C</TOOL-OUTPUT>
        <THOUGHT> I now have the weather information</THOUGHT>
        <FINAL-ANSWER> The weather in Paris is sunny with a temperature of 25°C.</FINAL-ANSWER>

        ## EXAMPLE WITHOUT TOOLS
        Question: Who wrote Romeo and Juliet?
        <THOUGHT> This is a literary knowledge question about Shakespeare</THOUGHT>
        <FINAL-ANSWER> William Shakespeare wrote Romeo and Juliet.</FINAL-ANSWER>

        Now, respond to the user's question using the format above.
        """
