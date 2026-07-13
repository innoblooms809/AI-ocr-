import json
from typing import List, Dict, Any

def format_to_json(qa_pairs: List[Dict[str, str]], raw_text: str = "") -> str:
    """
    Format extracted structured data into a JSON string.
    """
    output = {
        "metadata": {
            "total_questions": len(qa_pairs),
            "status": "success" if qa_pairs else "no_questions_extracted"
        },
        "qa_pairs": qa_pairs
    }
    
    if raw_text:
        output["raw_text"] = raw_text
        
    return json.dumps(output, indent=4, ensure_ascii=False)
