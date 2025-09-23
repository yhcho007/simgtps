"""Output filtering utilities:
- output_filter: blocks sensitive topics by simple heuristics
- mask_pii: mask patterns that look like account/card numbers

Extend these functions for production-grade filtering (NLP-based PII detectors, regexp sets, DLP systems).
"""
import re

SENSITIVE_PATTERNS = [r'\b\d{3}-\d{2}-\d{5}\b', r'\b\d{10,19}\b']


def output_filter(text: str) -> str:
    # Basic forbidden tokens check
    lowered = text.lower()
    if '비밀번호' in lowered or 'ssn' in lowered:
        return '민감한 정보는 제공할 수 없습니다.'
    # other safety rules can be added
    return text


def mask_pii(text: str) -> str:
    for p in SENSITIVE_PATTERNS:
        text = re.sub(p, '[REDACTED]', text)
    return text
