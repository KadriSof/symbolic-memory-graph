"""
Unified Prompt Templates for the Cogito Agent.
Optimized for token efficiency and strict schema alignment.
"""


class Prompts:
    """Unified prompt templates for the Cogito pipeline."""

    # Step 1: COMPREHEND
    @staticmethod
    def comprehend(query: str) -> str:
        """Unified comprehension prompt."""
        from .schemas import get_llm_format_instructions, ComprehensionSchema

        format_instructions = get_llm_format_instructions(ComprehensionSchema)

        return f"""
        You are a precise query analyzer for a cognitive agent.

        **CORE DIRECTIVE:** Choose the SIMPLEST path. 
        - REACT: Fast, cheap. Use for factual questions, simple tool calls, direct lookups.
        - COGITO: Complex, expensive. Use ONLY for relationships, contradictions, multi-step memory reasoning, or building mental models.
        *When in doubt, choose REACT.*

        **CRITICAL RULE:** You MUST extract ALL entities and relations from the input, regardless of the chosen path. Learning is mandatory.

        Original user query:
        ---
        {query}
        ---

        Return a JSON object matching this structure:
        {format_instructions}

        **EXAMPLES:**

        1. Simple query → REACT:
        {{
            "goal": "What is the capital of Tunisia?",
            "plan": ["Retrieve capital of Tunisia"],
            "user_intent": "ask",
            "modus_operandi": "REACT",
            "active_context": "question_answering",
            "entities": [
                {{"label": "Tunisia", "type": "country"}}
            ],
            "relations": [],
            "confidence": 0.95
        }}

        2. Complex query → COGITO:
        {{
            "goal": "Explain the relationship between Geralt and Yennefer",
            "plan": ["Retrieve Geralt", "Retrieve Yennefer", "Analyze relationship"],
            "user_intent": "ask",
            "modus_operandi": "COGITO",
            "active_context": "lore_building",
            "entities": [
                {{"label": "Geralt of Rivia", "type": "character"}},
                {{"label": "Yennefer of Vengerberg", "type": "character"}}
            ],
            "relations": [
                {{"source_label": "Geralt of Rivia", "target_label": "Yennefer of Vengerberg", "label": "loves", "direction": "asymmetric"}}
            ],
            "confidence": 0.9
        }}

        Return ONLY valid JSON.
        """

    # Step 3: CONSOLIDATE
    @staticmethod
    def consolidate(context: str, new_primitives: str, query: str) -> str:
        """Consolidation prompt."""
        from .schemas import get_llm_format_instructions, ConsolidationSchema

        format_instructions = get_llm_format_instructions(ConsolidationSchema)

        return f"""
        You are a knowledge consolidation engine. Compare current knowledge with new information and propose minimal, precise updates to the knowledge graph.

        Current knowledge (from the graph):
        {context}

        New information (extracted from the user query):
        {new_primitives}

        User query:
        {query}

        Return a JSON object matching this structure:
        {format_instructions}

        **CRITICAL INSTRUCTIONS:**
        1. **NO SELF-REFERENCES**: A relation's `source_label` and `target_label` MUST NOT be the same entity.
        2. **NO DUPLICATES**: Consolidate into a single, unique relation.
        3. **new_entities / new_relations**: ONLY include items that are TRULY new.
        4. **gaps**: MUST be a list of simple STRING questions (e.g., ["What is the capital of X?"]). DO NOT output objects.
        5. Return ONLY valid JSON.
        """

    # Step 4: REASON
    @staticmethod
    def reason(
        query: str, context: str, gaps: str, tool_results: str | None = None
    ) -> str:
        """Reasoning prompt."""
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
        3. What gaps must I fill?
        4. What is the final solution?

        **CRITICAL OUTPUT RULE:** 
        If you cannot answer the query with the provided context and lack critical information, you MUST explicitly state: "I need more information about [X]". 
        Otherwise, provide your final answer clearly after the word "Solution:".

        Format:
        [Your step-by-step reasoning...]

        Solution: [Your final answer OR "I need more information about X"]
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

        Keep the response concise, helpful, and friendly. Do not mention the graph or reasoning process unless directly relevant.
        """
