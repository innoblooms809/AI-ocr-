import re

def clean_markdown(text: str) -> str:
    """Clean and normalize the generated markdown text."""
    if not text:
        return ""
        
    # Standardize windows newlines
    text = text.replace("\r\n", "\n")
    
    # Strip leading/trailing whitespaces and double line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove any residual chat tags or generation prompts if leaked
    text = re.sub(r"<\|im_start\|>|<\|im_end\|>", "", text)
    text = re.sub(r"\[\s*IMAGE_PLACEHOLDER\s*\]", "", text, flags=re.IGNORECASE)
    
    return text.strip()

def strip_extra_whitespace(text: str) -> str:
    """Strips duplicate spacing in line segments while preserving markdown newlines."""
    lines = text.split("\n")
    cleaned_lines = [re.sub(r' +', ' ', line).strip() for line in lines]
    return "\n".join(cleaned_lines)
