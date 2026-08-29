"""
utils.py
Safe JSON parsing and small formatting helpers. Invalid JSON must never
crash the app.
"""

import json
import re
from typing import List, Optional, Tuple


def safe_parse_json(raw_text: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Strips accidental ```json fences / surrounding text and safely parses
    JSON. Returns (data, error) where exactly one of them is None.
    """
    if not raw_text:
        return None, "Empty response from the model."

    text = raw_text.strip()
    text = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()

    # If there's leading/trailing prose around the JSON object, try to
    # isolate the outermost { ... } block.
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)

    try:
        data = json.loads(text)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"Could not parse the model's JSON response: {e}"


def format_symptom_summary(symptoms: List[str], free_text: str) -> str:
    parts = list(symptoms) if symptoms else []
    if free_text and free_text.strip():
        parts.append(free_text.strip())
    return ", ".join(parts) if parts else "None reported"


def urgency_badge(level: str) -> str:
    from src.config import URGENCY_COLORS
    key = (level or "").upper()
    icon = URGENCY_COLORS.get(key, "\u26AA")  # white circle fallback
    return f"{icon} {key or 'UNKNOWN'}"
