"""
Ad creative compliance checker for GPR ADX.

Provides a two-layer compliance verification system:
  Layer 1 — Rule-based keyword scanning against a blocklist of
            prohibited advertising terms (Chinese + English).
  Layer 2 — LLM-based semantic compliance re-check for edge cases
            that the keyword scanner may miss.

Both layers run on the agent side-path and are NEVER invoked on
the synchronous RTB hot path.
"""

from typing import List, Tuple


# ---------------------------------------------------------------------------
# Prohibited terms — advertising regulations (CN + international)
# ---------------------------------------------------------------------------
SENSITIVE_WORDS: set = {
    # Chinese Advertising Law  — absolute / superlative claims
    "最", "第一", "唯一", "顶级", "极致", "完美", "绝佳",
    "国家级", "世界级", "最高级", "最佳", "顶级", "极品",
    "首个", "首家", "全网第一", "独一无二",
    # Medical / efficacy claims
    "治愈", "根治", "无副作用", "药到病除", "特效",
    "包治百病", "神药", "一疗程除根",
    # Wealth / earnings claims
    "包赚", "稳赚", "月入百万", "日进斗金",
    # English advertising  — absolute / superlative claims
    "guaranteed", "100%", "miracle", "cure-all",
    "risk-free", "no-risk", "instant", "magic",
    "best", "number one", "#1", "unbeatable",
    "scientifically proven", "perfect",
    # Spam / deceptive patterns
    "click here", "act now", "limited time",
    "exclusive offer", "secret",
    # Financial claims
    "get rich", "earn money fast", "double your",
    # Weight loss / body image
    "lose weight fast", "miracle diet", "fat burner",
}


def check_compliance(title: str, description: str) -> Tuple[bool, List[str]]:
    """Layer-1 rule-based compliance check.

    Scans *title* and *description* for prohibited terms defined
    in ``SENSITIVE_WORDS``.  Matching is case-insensitive.

    Returns:
        ``(passed: bool, violations: list[str])`` where *passed* is
        ``True`` when no violations are found.
    """
    violations: List[str] = []
    combined = f"{title} {description}"

    # Case-insensitive lookup for English; exact substring for Chinese
    lower_text = combined.lower()
    for word in SENSITIVE_WORDS:
        # Chinese words are matched directly; English words against
        # the lower-cased version of the text.
        needle = word.lower() if any(c.isascii() for c in word) else word
        if needle in (combined if not needle.isascii() else lower_text):
            violations.append(word)

    passed = len(violations) == 0
    return passed, violations


def llm_compliance_check(
    title: str,
    description: str,
    llm,  # LangChain ChatOpenAI instance (injected)
) -> Tuple[bool, List[str]]:
    """Layer-2 LLM-based semantic compliance re-check.

    Uses a small prompt to have the LLM evaluate whether the
    creative contains misleading, exaggerated, or otherwise
    non-compliant language that the keyword scanner may have
    missed.

    Returns:
        ``(passed: bool, violations: list[str])``
    """
    prompt = (
        "You are an advertising compliance reviewer. "
        "Review the following ad creative and determine if it "
        "contains any misleading, exaggerated, false, or "
        "otherwise non-compliant claims.\n\n"
        f"Title: {title}\n"
        f"Description: {description}\n\n"
        "Respond with a JSON object: "
        '{"passed": true/false, "violations": ["..."], "reason": "..."}'
    )

    try:
        response = llm.invoke(prompt)
        import json

        content = response.content if hasattr(response, "content") else str(response)
        result = json.loads(content)
        passed = result.get("passed", True)
        violations = result.get("violations", [])
        return passed, violations
    except Exception:
        # If the LLM is unavailable or returns malformed JSON we
        # fall back to the rule-based result.
        return True, []
