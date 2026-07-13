from typing import List, Dict

def extract_answers(text: str, questions: List[str]) -> List[Dict[str, str]]:
    """
    Pairs questions with their corresponding answers by scanning the source text.
    Assumes that the answer to a question immediately follows it until the next question starts.
    """
    if not questions:
        return []
        
    qa_pairs = []
    
    # Simple algorithm: Find the indices of all questions in the text,
    # then slice the text between question N and question N+1 to get the answer.
    indices = []
    for q in questions:
        idx = text.find(q)
        if idx != -1:
            indices.append((idx, q))
            
    # Sort indices by their position in the text
    indices.sort()
    
    for i in range(len(indices)):
        start_idx = indices[i][0] + len(indices[i][1])
        end_idx = indices[i+1][0] if i + 1 < len(indices) else len(text)
        
        answer_block = text[start_idx:end_idx].strip()
        
        # Clean up leading 'A:', 'Answer:', or symbols from the answer
        if answer_block.lower().startswith("a:"):
            answer_block = answer_block[2:].strip()
        elif answer_block.lower().startswith("answer:"):
            answer_block = answer_block[7:].strip()
        elif answer_block.startswith(":") or answer_block.startswith("-") or answer_block.startswith("："):
            answer_block = answer_block[1:].strip()
            
        qa_pairs.append({
            "question": indices[i][1],
            "answer": answer_block
        })
        
    return qa_pairs
