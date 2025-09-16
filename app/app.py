from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Tuple
import re
import string

app = FastAPI(title="LLM Prompt Condenser", version="1.0.0")

# ------------------------------
# Input/Output models
# ------------------------------
class CondenseIn(BaseModel):
    llm_prompt: List[List[str]]
    keep_sections: bool = False
    include_notes: bool = False
    max_bullets: int = 0

class CondenseOut(BaseModel):
    concise_llm_prompt: List[List[str]]
    bullets: List[str]
    notes: List[str]
    stats: Dict[str, int]

# ------------------------------
# Heuristics & helpers (unchanged)
# ------------------------------

#... [same code for helpers / heuristics as before] ...

# Clause/keyword patterns indicating "directive" lines to keep
DIRECTIVE_PATTERNS = [
    r"\b(must|shall|should)\b",
    r"\b(do\s+not|don’t|never|avoid|forbid|forbidden|prohibit)\b",
    r"^(?:-|\*|•)\s+",
    r"^\s*\d+\s*[\.\)]\s+",
    r"^rule\s*\d+",
    r"\b(use|replace|prefer|ensure|rewrite|refactor|guard|escape|return)\b",
    r"\b(select\s+single\b|\bup\s+to\s+1\s+rows\b|\border\s+by\b|\bjoin\b|\bexists\b|\bsubquer|for all entries\b|\bcds\b)",
    r"\bdraft\b|\bisactiveentity\b",
    r"\bstrict\s+json\b|\bonly\s+json\b",
]

DROP_PATTERNS = [
    r"^\s*```", r"^\s*</?retrieved_rules>", r"^\s*before:?", r"^\s*after:?",
    r"^\s*why:?", r"^\s*scope:?", r"^\s*notes:?", r"^\s*pattern:?", r"^\s*preferred:?|^\s*alternative:?|^\s*fallback:?",
    r"^\s*example", r"^\s*examples",
]

PRIORITY: List[Tuple[str, int]] = [
    (r"\bstrict\s+json\b|\bonly\s+json\b",                 100),
    (r"\bselect\s+single\b.*\border\s+by\b",               97),
    (r"\bup\s+to\s+1\s+rows\b",                            95),
    (r"@-?escape|\bhost\s+variable|\bINTO\b.*@|\bWHERE\b.*@", 92),
    (r"\bexplicit\s+field\s+list|\bselect\s+\*",           90),
    (r"\bjoin\b|\bexists\b|\bsubquery|\bfor all entries\b",88),
    (r"\bcds\b|nsdm|matdoc|v_konv",                        85),
    (r"\bdraft\b|isactiveentity",                          82),
    (r"\bprimary\s+key|\bunique\s+index",                  80),
    (r"\bsort\b|\bbinary\s+search|\bdeduplicate",          70),
]

PUNCT_TRANSLATE = str.maketrans("", "", string.punctuation)

def _strip_code_fences(text: str) -> str:
    return re.sub(r"(?s)```.*?```", "", text)

def _split_lines(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [ln for ln in text.split("\n")]

def _looks_directive(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    for pat in DROP_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            return False
    for pat in DIRECTIVE_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            return True
    return False

def _clean_line(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^(?:-|\*|•)\s+", "", s)
    s = re.sub(r"^\d+\s*[\.\)]\s+", "", s)
    s = re.sub(r'"\s*Added\s+By\s+Pwc[0-9\-]*\s*$', "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def _norm_for_dedupe(line: str) -> str:
    s = line.lower()
    s = s.translate(PUNCT_TRANSLATE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _priority(line: str) -> int:
    for pat, weight in PRIORITY:
        if re.search(pat, line, flags=re.IGNORECASE):
            return weight
    return 60  # default mid priority

def _is_heading(line: str) -> bool:
    return bool(re.match(r"^\s*rule\s*\d+|^\s*[A-Z][A-Za-z0-9 ]+—", line.strip(), flags=re.IGNORECASE))

# ------------------------------
# Modified: condense_text accepts llm_prompt as List[List[str]]!
# ------------------------------
def condense_text(
    src_table: List[List[str]],
    keep_sections: bool,
    include_notes: bool,
    max_bullets: int,
) -> CondenseOut:
    # Flatten [["a", "b"], ["c"]] ⇒ ["a b", "c"]
    flat_lines = [" ".join([c for c in row if c is not None]) for row in src_table]
    original = "\n".join(flat_lines)
    no_code = _strip_code_fences(original)
    lines = _split_lines(no_code)

    directive_raw: List[str] = []
    non_directive: List[str] = []

    for ln in lines:
        if _looks_directive(ln):
            directive_raw.append(ln)
        else:
            non_directive.append(ln.strip())

    seen = set()
    kept: List[str] = []
    for ln in directive_raw:
        cl = _clean_line(ln)
        if not cl:
            continue
        if _is_heading(cl) and not keep_sections:
            continue
        norm = _norm_for_dedupe(cl)
        if norm and norm not in seen:
            seen.add(norm)
            kept.append(cl)

    kept_sorted = sorted(kept, key=lambda x: (-_priority(x), len(x)))

    if max_bullets and max_bullets > 0:
        kept_sorted = kept_sorted[:max_bullets]

    bullets = [f"- {x}" for x in kept_sorted]

    header = (
        "Follow these rules exactly. If any bullets conflict, prioritize correctness and S/4HANA compatibility. Return ONLY what is requested."
    )
    # For "table" output, split by "\n" into cell per line
    all_lines = [header] + bullets  # "" blank line
    concise_llm_prompt_table = [[l] for l in all_lines]  # Each line is a row with one cell

    notes_out: List[str] = []
    if include_notes:
        tmp = []
        for s in non_directive:
            if not s:
                continue
            if re.search("|".join(DROP_PATTERNS), s, flags=re.IGNORECASE):
                continue
            if len(s) > 140:
                continue
            tmp.append(s)
        seen_n = set()
        for s in tmp:
            n = _norm_for_dedupe(s)
            if n and n not in seen_n:
                seen_n.add(n)
                notes_out.append(s)

    return CondenseOut(
        concise_llm_prompt=concise_llm_prompt_table,
        bullets=bullets,
        notes=notes_out,
        stats={
            "input_chars": len(original),
            "input_lines": len(lines),
            "candidate_directives": len(directive_raw),
            "kept_bullets": len(bullets),
            "notes": len(notes_out),
        },
    )

# ------------------------------
# API
# ------------------------------
@app.post("/condense", response_model=List[CondenseOut])
async def api_condense(llm_prompt: List[str]):
    """
    Accepts a bare array of strings as input, 
    returns a list with a single object, 
    where the concise_llm_prompt is a table (list of 1-item lists).
    """
    # Convert to the expected "table" format (List[List[str]])
    llm_prompt_table = [[line] for line in llm_prompt]
    # Use your existing CondenseIn, with default options.
    body = CondenseIn(llm_prompt=llm_prompt_table)
    out = condense_text(
        src_table=body.llm_prompt,
        keep_sections=body.keep_sections,
        include_notes=body.include_notes,
        max_bullets=body.max_bullets,
    )
    # Always return a list per the contract
    return [out]