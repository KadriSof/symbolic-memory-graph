# python/src/memory_agent/core/prompts.py

from __future__ import annotations

import json
from typing import Any

from ...tools import Tool

from .schemas import (
    ComprehensionSchema,
    ConsolidationSchema,
    ReasoningSchema,
    get_llm_format_instructions,
)


class Prompts:
    """Prompt factory for Cogito cognitive procedures."""

    # COMPREHENSION
    @staticmethod
    def comprehend(query: str) -> str:
        format_instructions = get_llm_format_instructions(ComprehensionSchema)

        return f"""
        You are the COMPREHENSION subsystem of Cogito.

        Your job is to transform the user's input into a structured representation.

        ## USER INPUT:
        {query}

        ## OBJECTIVES:

        1. Determine the user's actual goal.
        2. Produce a useful high-level plan when appropriate.
        3. Identify the user's intent.
        4. Classify the general context.
        5. Create a brief summary of the context.
        6. Extract ALL identifiable entities.
        7. Extract ALL meaningful relations between those entities.
        8. Score the overall confidence.

        **IMPORTANT**:

        Comprehension does NOT decide whether Cogito should:
        - reply,
        - request clarification,
        - or use a tool.

        That decision belongs exclusively to the REASON subsystem.

        ## ENTITY EXTRACTION:

        Extract entities even when they are not necessary for answering the
        question. Learning and memory formation are separate from routing.

        ## RELATION EXTRACTION:

        Extract explicit or strongly implied relationships between entities.

        **Do not invent relationships.**

        {format_instructions}
        """.strip()

    # REASON
    @staticmethod
    def reason(
        query: str,
        context: str,
        gaps: str = "None",
        tool_results: str | None = None,
        clarification_response: str | None = None,
        tools: dict[str, Tool] | None = None,
    ) -> str:

        format_instructions = get_llm_format_instructions(ReasoningSchema)

        tool_results_text = (
            tool_results if tool_results else "No tool observations are available."
        )

        clarification_text = (
            clarification_response
            if clarification_response
            else "No clarification response is pending."
        )

        if tools:
            available_tools = "\n\n".join(repr(tool) for tool in tools.values())

        else:
            available_tools = "No tools are available."

        return f"""
        You are the REASON subsystem of Cogito.

        Your responsibility is to determine the NEXT COGNITIVE ACTION.

        You must choose exactly ONE:

        REPLY
        REQUEST
        REACT

        The decision must be based on the information currently available.

        ## USER REQUEST

        {query}

        ## RECALLED KNOWLEDGE

        {context}

        ## KNOWN INFORMATION GAPS

        {gaps}

        ## PREVIOUS TOOL OBSERVATIONS

        {tool_results_text}

        ## PREVIOUS CLARIFICATION RESPONSE

        {clarification_text}

        ## AVAILABLE TOOLS

        {available_tools}

        ## DECISION POLICY

        1. REPLY

        Choose REPLY when you have enough information to answer the user's request.

        Use REPLY for:
        - general knowledge questions,
        - Creative writing,
        - explanations,
        - definitions,
        - known historical/literary/factual information,
        - reasoning that can be completed from available knowledge,
        - questions where memory and internal knowledge are sufficient.

        **Do NOT use REACT merely because a tool exists.**

        **Do NOT use REQUEST merely because you are uncertain.**

        2. REQUEST

        Choose REQUEST ONLY when essential information is missing from the USER
        and that information cannot be obtained by an available tool.

        Examples:
        - "Book me a flight" with no destination.
        - "Compare these two systems" when one system is not identified.
        - "Send this to them" when the recipient is unknown.

        If an available tool can obtain the missing information, choose REACT instead.

        3. REACT

        Choose REACT when an available tool can obtain information or perform an
        operation required to answer the request.

        Examples:
        - current weather,
        - current stock price,
        - current exchange rate,
        - filesystem inspection,
        - database lookup,
        - calculation when a calculator tool is appropriate,
        - external API information.

        CRITICAL DISTINCTION:

        Missing external/current information + suitable tool
        => REACT

        Missing user-provided information + no tool can obtain it
        => REQUEST

        Sufficient information
        => REPLY

        ## AFTER TOOL USE

        Tool observations are NOT automatically the final answer.

        After REACT obtains an observation, Cogito returns to REASON.

        At the next REASON step:

        - If the observation provides enough information => REPLY.
        - If another tool is required => REACT.
        - If the user must provide missing information => REQUEST.

        Never jump directly from REACT to REPLY without evaluating the result.

        ## CLARIFICATION

        If a clarification response is present, treat it as additional information
        provided by the user for the existing task.

        Do not request the same information again unless the response genuinely
        fails to resolve the missing requirement.

        ## OUTPUT

        {format_instructions}

        Additional decision requirements:

        - `decision` MUST be exactly one of:
        REPLY
        REQUEST
        REACT

        - For REPLY:
        - provide `answer`
        - `clarification_request` should be empty
        - `tool_intent` should be empty

        - For REQUEST:
        - provide `clarification_request`
        - explain the missing user information in `gaps`
        - `answer` should normally be empty

        - For REACT:
        - provide `tool_intent`
        - identify what information/action the tool should provide
        - `answer` should normally be empty

        Do not return multiple decisions.
        """.strip()

    # CONSOLIDATION
    @staticmethod
    def consolidate(
        query: str,
        answer: str,
        entities: list[Any],
        relations: list[Any],
    ) -> str:

        format_instructions = get_llm_format_instructions(ConsolidationSchema)

        entity_data = [
            {
                "label": entity.label,
                "type": entity.type,
                "confidence": entity.confidence,
            }
            for entity in entities
        ]

        relation_data = [
            {
                "source": relation.source_label,
                "target": relation.target_label,
                "label": relation.label,
                "direction": relation.direction,
                "confidence": relation.confidence,
            }
            for relation in relations
        ]

        return f"""
        You are the CONSOLIDATION subsystem of Cogito.

        Your task is to identify durable knowledge learned from the interaction.

        USER QUERY:
        {query}

        ASSISTANT ANSWER:
        {answer}

        EXTRACTED ENTITIES:
        {json.dumps(entity_data, ensure_ascii=False, indent=2)}

        EXTRACTED RELATIONS:
        {json.dumps(relation_data, ensure_ascii=False, indent=2)}

        RULES:

        1. Preserve only useful durable knowledge.
        2. Do not invent entities.
        3. Do not invent relations.
        4. Avoid duplicate knowledge.
        5. Never create self-referential relations unless explicitly meaningful.
        6. Preserve confidence information.
        7. Gaps must be concise strings.
        8. Conflicts must describe actual conflicting knowledge.
        9. Do not store transient reasoning as durable knowledge.

        {format_instructions}
        """.strip()
