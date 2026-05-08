from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Sequence

BOXED_FEWSHOT_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
    "Example:\n"
    "User: If 2 apples plus 3 apples equals how many apples?\n"
    "Assistant: We add 2 + 3 = 5, so the final answer is \\boxed{5}.\n\n"
    "Always finish your response by repeating only the final answer inside a single \\boxed{}."
)


def enforce_boxed_system_prompt(messages: Any) -> Any:
    if not isinstance(messages, Sequence):
        return messages
    try:
        msgs: List[dict] = [dict(m) for m in messages]
    except Exception:
        return messages

    if not msgs:
        return [{"role": "system", "content": BOXED_FEWSHOT_SYSTEM_PROMPT}]

    first = msgs[0]
    if first.get("role") == "system":
        content = first.get("content", "")
        if "Always finish your response" in content:
            return msgs
        first["content"] = BOXED_FEWSHOT_SYSTEM_PROMPT
    else:
        msgs.insert(0, {"role": "system", "content": BOXED_FEWSHOT_SYSTEM_PROMPT})
    return msgs
