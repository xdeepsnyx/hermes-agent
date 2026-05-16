"""Rule modules for tool_output_reducer.

Each rule exposes a single function ``(text: str) -> str`` that returns
the transformed text (or the original if no change applies). Rules must
be idempotent and never raise on well-formed string input.
"""
