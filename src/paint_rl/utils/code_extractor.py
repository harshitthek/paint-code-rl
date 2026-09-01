"""Resilient multi-stage JavaScript code extractor for generative model completions.

Handles closed fences, unclosed fences (truncated mid-generation), thinking tags,
trailing unclosed string literals, and missing function closure braces.
"""
import re


def _count_structural_braces(code: str) -> tuple[int, int]:
    """Count structural curly braces outside comments and string literals."""
    # Remove single-line comments
    cleaned = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    # Remove multi-line comments
    cleaned = re.sub(r'/\*[\s\S]*?\*/', '', cleaned)
    # Remove strings: double-quoted, single-quoted, and template literals
    cleaned = re.sub(r'"(?:\\.|[^"\\])*"', '""', cleaned)
    cleaned = re.sub(r"'(?:\\.|[^'\\])*'", "''", cleaned)
    cleaned = re.sub(r'`(?:\\.|[^`\\])*`', '``', cleaned)
    return cleaned.count("{"), cleaned.count("}")


def _sanitize_trailing_truncation(code: str) -> str:
    """Sanitize code that was truncated mid-token or mid-statement by token limits."""
    if not code:
        return ""
    
    lines = code.split("\n")
    if not lines:
        return code
    
    # Check the last line for obvious truncation (unclosed string literal, trailing dot, unclosed parens)
    last_line = lines[-1].strip()
    if last_line:
        # Check for odd number of single/double quotes on the last line (unclosed literal like `brush.stroke('#ff3`)
        single_quotes = last_line.count("'") - last_line.count(r"\'")
        double_quotes = last_line.count('"') - last_line.count(r'\"')
        
        if (single_quotes % 2 != 0) or (double_quotes % 2 != 0) or last_line.endswith("(") or last_line.endswith(",") or last_line.endswith("."):
            # Remove the truncated last line
            lines.pop()
            code = "\n".join(lines).strip()
    
    # Balance unclosed structural curly braces
    open_braces, close_braces = _count_structural_braces(code)
    if open_braces > close_braces:
        code += "\n" + ("}\n" * (open_braces - close_braces))
        
    return code.strip()


def robust_extract_js_code(raw_text: str) -> str:
    """Extract clean, executable JavaScript from model completions under all edge cases.
    
    Stages:
    1. Strip <think>...</think> reasoning blocks.
    2. Extract closed ```javascript ... ``` markdown blocks.
    3. Extract unclosed ```javascript ... blocks (when generation hits token limits mid-code).
    4. Fallback search for raw `function setup()` entry point with postamble stripping.
    5. Sanitize trailing truncation and balance missing closing braces.
    """
    if not raw_text:
        return ""
    
    if isinstance(raw_text, list):
        if len(raw_text) > 0 and isinstance(raw_text[-1], dict) and "content" in raw_text[-1]:
            raw_text = raw_text[-1]["content"]
        else:
            raw_text = str(raw_text)
    
    text = str(raw_text)
    
    # 1. Strip reasoning/thinking tags (e.g. DeepSeek/Qwen thinking mode)
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()
    
    # 2. Closed markdown code fences (supports js, javascript, p5, p5js, or untagged)
    closed_fence_match = re.search(
        r'```(?:javascript|js|p5js|p5)?\s*\n([\s\S]*?)```', 
        text, 
        flags=re.IGNORECASE
    )
    if closed_fence_match:
        code = closed_fence_match.group(1).strip()
        if code:
            return _sanitize_trailing_truncation(code)
            
    # 3. Unclosed markdown code fence (generation truncated mid-code without closing ```)
    unclosed_fence_match = re.search(
        r'```(?:javascript|js|p5js|p5)?\s*\n([\s\S]*)$', 
        text, 
        flags=re.IGNORECASE
    )
    if unclosed_fence_match:
        code = unclosed_fence_match.group(1).strip()
        # Clean any trailing partial backticks
        code = re.sub(r'`+$', '', code).strip()
        if code:
            return _sanitize_trailing_truncation(code)
            
    # 4. Fallback: Search for raw p5.js entry point (function setup / function draw)
    setup_match = re.search(r'(function\s+setup\s*\(\)[\s\S]*)', text)
    if setup_match:
        code = setup_match.group(1).strip()
        # Strip trailing markdown fences if present
        code = re.sub(r'```.*$', '', code, flags=re.DOTALL).strip()
        return _sanitize_trailing_truncation(code)
        
    return _sanitize_trailing_truncation(text.strip())
