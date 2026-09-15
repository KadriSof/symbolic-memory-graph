# src/memory_agent/llm/summarization.py
from .base import BaseLLM
from typing import Any


def summarize_conversation(
    messages: list[dict[str, Any]],
    llm: BaseLLM,
    max_tokens: int = 500,
) -> str:
    """
    Summarize a conversation using the LLM.

    Args:
        messages: List of conversation messages (e.g., [{"role": "user", "content": "..."}, ...]).
        llm: The LLM instance to use for summarization.
        max_tokens: Maximum tokens for the summary.

    Returns:
        A concise summary of the conversation.
    """
    # Prepare the prompt for summarization
    prompt = """
    Please summarize the following conversation concisely while preserving key details.
    Focus on the most important points, decisions, and context.
    Keep the summary under {max_tokens} tokens.

    Conversation:
    {conversation}
    """.format(
        max_tokens=max_tokens,
        conversation="\n".join(f"{msg['role']}: {msg['content']}" for msg in messages),
    )

    # Call the LLM to generate the summary
    summary = llm.chat(
        messages=[{"role": "system", "content": prompt}],
    )

    return summary
