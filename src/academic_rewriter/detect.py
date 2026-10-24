"""Transparent heuristic detector prototype. It is not a trained style classifier."""
import re
PATTERNS=[(r'\ba lot of\b','Consider a more precise quantifier'),(r'\b(?:very|really)\b','Check unnecessary intensification'),(r'\b(?:looked at|got|stuff|things)\b','Check conversational or vague wording')]
def detect(text):
    issues=[dict(start=m.start(),end=m.end(),text=m.group(),reason=reason) for pattern,reason in PATTERNS for m in re.finditer(pattern,text,re.I)]
    return {'revise':bool(issues),'issues':sorted(issues,key=lambda x:x['start']),'method':'heuristic prototype','warning':'A negative result does not establish acceptable academic style.'}
