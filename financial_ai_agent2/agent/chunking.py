import re
from typing import List

def chunk_korean_text(text: str, max_chars: int = 800, overlap: int = 128) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r'(?<=[.?!。！？\n])\s*', text)
    chunks = []
    cur = ""
    for s in sentences:
        if not s: continue
        if len(cur) + len(s) <= max_chars:
            cur = (cur + " " + s).strip() if cur else s.strip()
        else:
            if cur: chunks.append(cur)
            if len(s) <= max_chars:
                cur = s.strip()
            else:
                start = 0
                while start < len(s):
                    part = s[start:start+max_chars]
                    chunks.append(part.strip())
                    start += max_chars - overlap
                cur = ""
    if cur: chunks.append(cur)
    return [c for c in chunks if c]
