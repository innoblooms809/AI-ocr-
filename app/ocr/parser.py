import re
from typing import Dict, Any, List

def parse_ocr_text(text: str) -> Dict[str, Any]:
    """
    Parses structural patterns from VLM output text.
    For example, extracting table tags, code blocks, or inline formulas.
    """
    result = {
        "raw_text": text,
        "tables": [],
        "formulas": [],
        "markdown": text
    }
    
    # Simple regex parsing for markdown tables
    table_matches = re.findall(r"(\|.*\|?\n\|[-:| ]*\|?\n(?:\|.*\|?\n)+)", text)
    if table_matches:
        result["tables"] = [table.strip() for table in table_matches]
        
    # Simple regex parsing for LaTeX formulas
    formula_matches = re.findall(r"(\$\$[\s\S]*?\$\$|\$.*?\$)", text)
    if formula_matches:
        result["formulas"] = [formula.strip() for formula in formula_matches]
        
    return result
