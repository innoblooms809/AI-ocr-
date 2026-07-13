import re
from typing import List

def extract_questions(text: str) -> List[str]:
    """
    Scans the OCR text and isolates question blocks.
    Looks for markers like 'Q:', 'Question:', or numbered items (e.g., '1. ', 'Q1.').
    """
    # Regex to catch:
    # 1. "Q: ..." or "Question 1: ..."
    # 2. Numbered lines like "1. What is..." or "2) Why..."
    question_patterns = [
        r"(?:(?:Question|Q|q)\s*(?:\d+)?\s*[:.-]\s*)(.+)",  # Q: What is... or Question 1: What is...
        r"(?:\n|^)(?:\d+[\.\)])\s*(.+)"                      # 1. What is... or 1) What is...
    ]
    
    questions = []
    
    # Simple line-by-line or paragraph analysis
    lines = text.split("\n")
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        matched = False
        for pattern in question_patterns:
            match = re.match(pattern, line_stripped)
            if match:
                questions.append(match.group(1).strip())
                matched = True
                break
                
    # If no matches are found, return non-empty lines starting with a capital letter and ending with a question mark
    if not questions:
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.endswith("?") or line_stripped.endswith("？"):
                questions.append(line_stripped)
                
    return questions
