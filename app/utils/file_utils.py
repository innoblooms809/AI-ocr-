import os
import json
from typing import Any, Dict

def ensure_dir(path: str) -> str:
    """Ensure that the directory path exists."""
    os.makedirs(path, exist_ok=True)
    return path

def save_json(data: Any, filepath: str) -> None:
    """Save data as JSON to the specified path."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_text(text: str, filepath: str) -> None:
    """Save plain text or markdown to the specified path."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
