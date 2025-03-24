"""Utility functions for processing messages for different LLM client formats."""
import json
from typing import Dict, Any, List, Tuple, Optional

def msg_dict_to_oai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert message dictionaries to OpenAI format."""
    return messages

def msg_dict_to_anthropic(messages: List[Dict[str, Any]], use_cache: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert message dictionaries to Anthropic format."""
    return messages, messages

def parse_json_string(content: str) -> Optional[Dict[str, Any]]:
    """Parse JSON string safely with fallback strategies."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try parsing from code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        return None 