from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional
import warnings

try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0
except Exception:  # pragma: no cover
    detect = None
_LANG_WARNED = False


DEVANAGARI = (0x0900, 0x097F)
GURMUKHI = (0x0A00, 0x0A7F)
GUJARATI = (0x0A80, 0x0AFF)
TAMIL = (0x0B80, 0x0BFF)
TELUGU = (0x0C00, 0x0C7F)
KANNADA = (0x0C80, 0x0CFF)
MALAYALAM = (0x0D00, 0x0D7F)
THAI = (0x0E00, 0x0E7F)
BENGALI = (0x0980, 0x09FF)
ARABIC = (0x0600, 0x06FF)
HANGUL = (0xAC00, 0xD7AF)
CJK = (0x4E00, 0x9FFF)

SCRIPT_MAP = {
    "hi": [DEVANAGARI],
    "mr": [DEVANAGARI],
    "ne": [DEVANAGARI],
    "bn": [BENGALI],
    "pa": [GURMUKHI],
    "gu": [GUJARATI],
    "ta": [TAMIL],
    "te": [TELUGU],
    "kn": [KANNADA],
    "ml": [MALAYALAM],
    "th": [THAI],
    "ur": [ARABIC],
    "ar": [ARABIC],
    "fa": [ARABIC],
    "ko": [HANGUL],
    "zh": [CJK],
    "ja": [CJK],
    "vi": [],  # Latin, rely on langdetect
    "de": [],
    "sw": [],
    "pt": [],
    "it": [],
    "fi": [],
    "ru": [(0x0400, 0x04FF)],
}


def extract_response_text(output: str) -> str:
    if not output:
        return ""
    text = output
    if "</think>" in text:
        text = text.split("</think>")[-1]
    lower = text.lower()
    markers = [
        "<|im_start|>assistant",
        "<|assistant|>",
        "\nassistant:",
        "\nassistant",
    ]
    last_idx = -1
    last_len = 0
    for marker in markers:
        idx = lower.rfind(marker)
        if idx > last_idx:
            last_idx = idx
            last_len = len(marker)
    if last_idx != -1:
        text = text[last_idx + last_len :]
    return text.strip()


def extract_user_prompt(prompt: Any) -> str:
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") in {"user", "human"}:
                return str(msg.get("content", "")).strip()
    if isinstance(prompt, str):
        return prompt.strip()
    return ""


def _compare(value: float, relation: Optional[str], target: Optional[float]) -> bool:
    if target is None:
        return False
    try:
        target_val = float(target)
    except (TypeError, ValueError):
        return False
    relation = (relation or "equal").lower()
    if relation in {"at least", ">=", "greater or equal"}:
        return value >= target_val
    if relation in {"at most", "<=", "no more than"}:
        return value <= target_val
    if relation in {"less than", "<"}:
        return value < target_val
    if relation in {"exactly", "equal", "equals", "=="}:
        return value == target_val
    return value == target_val


WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"[.!?]+")
HIGHLIGHT_RE = re.compile(r"\*{1,2}([^*\n]+)\*{1,2}")
PLACEHOLDER_RE = re.compile(r"\[[^\]]+\]")
BULLET_RE = re.compile(r"^\s*[\-\*\u2022]\s+", re.MULTILINE)
LETTER_RE_CACHE: Dict[tuple, re.Pattern] = {}


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _sentence_count(text: str) -> int:
    sentences = [seg.strip() for seg in SENTENCE_RE.split(text) if seg.strip()]
    return len(sentences)


def _paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _count_highlights(text: str) -> int:
    return len(HIGHLIGHT_RE.findall(text))


def _count_placeholders(text: str) -> int:
    return len(PLACEHOLDER_RE.findall(text))


def _count_bullets(text: str) -> int:
    return len(BULLET_RE.findall(text))


def _count_letter(text: str, letter: str) -> int:
    if not letter:
        return 0
    pattern = LETTER_RE_CACHE.get((letter, False))
    if pattern is None:
        pattern = re.compile(re.escape(letter), re.IGNORECASE)
        LETTER_RE_CACHE[(letter, False)] = pattern
    return len(pattern.findall(text))


def _count_keyword(text: str, keyword: str) -> int:
    if not keyword:
        return 0
    pattern = LETTER_RE_CACHE.get((keyword, True))
    if pattern is None:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        LETTER_RE_CACHE[(keyword, True)] = pattern
    return len(pattern.findall(text))


def _contains_script(text: str, ranges: Iterable[tuple]) -> bool:
    for ch in text:
        code = ord(ch)
        for start, end in ranges:
            if start <= code <= end:
                return True
    return False


PUNCT_STRIP = " \t\r\n\"'`.,!?;:-()[]{}"


def normalize_text_answer(text: Optional[str]) -> str:
    if not text:
        return ""
    normalized = " ".join(text.strip().split())
    normalized = normalized.strip(PUNCT_STRIP)
    return normalized.lower()


def text_answer_matches(output: str, ground_truth: str, aliases: Optional[List[str]] = None) -> bool:
    if not ground_truth:
        return False
    response_text = extract_response_text(output)
    norm_pred_full = normalize_text_answer(response_text)
    if not norm_pred_full:
        return False
    lines = [line for line in response_text.strip().splitlines() if line.strip()]
    norm_pred_last = normalize_text_answer(lines[-1]) if lines else norm_pred_full

    def _match(norm_candidate: str) -> bool:
        if not norm_candidate:
            return False
        return norm_candidate in norm_pred_full or norm_pred_full in norm_candidate or norm_candidate in norm_pred_last

    candidates = [ground_truth] + list(aliases or [])
    for cand in candidates:
        if _match(normalize_text_answer(cand)):
            return True
    return False


def _check_language(text: str, target_lang: str) -> bool:
    cleaned = re.sub(r"[^A-Za-z\u0080-\uFFFF]+", " ", text)
    if not cleaned.strip():
        return False
    target_lang = (target_lang or "").lower()
    if detect:
        snippet = cleaned
        if len(snippet) < 20 and len(cleaned) >= 20:
            snippet = cleaned[:200]
        try:
            detected = detect(snippet)
            if detected == target_lang:
                return True
        except (LangDetectException, ValueError):
            pass
    ranges = SCRIPT_MAP.get(target_lang)
    if ranges:
        return _contains_script(text, ranges)
    global _LANG_WARNED
    if detect is None and not _LANG_WARNED:
        warnings.warn(
            f"langdetect not installed; cannot verify language instruction for '{target_lang}'.",
            RuntimeWarning,
            stacklevel=2,
        )
        _LANG_WARNED = True
    return False


def _count_uppercase_words(text: str) -> int:
    words = WORD_RE.findall(text)
    return sum(1 for w in words if len(w) > 1 and w.isupper())


def _check_postscript(text: str, marker: Optional[str]) -> bool:
    if not marker:
        return False
    marker_lower = marker.lower()
    lower_text = text.lower()
    idx = lower_text.rfind(marker_lower)
    if idx == -1:
        return False
    # Expect marker near the end (last third of the content)
    return idx >= len(lower_text) * 2 / 3


def _check_constrained_response(text: str, prompt_text: str) -> bool:
    options = re.findall(r"'([^']+)'", prompt_text)
    if not options:
        return False
    normalized = text.strip()
    return any(normalized == opt or normalized.rstrip(".") == opt.rstrip(".") for opt in options)


def evaluate_instruction(text: str, instruction: Dict[str, Any], prompt_text: str) -> bool:
    inst_id = instruction.get("instruction_id", "")
    kwargs = instruction.get("kwargs") or {}

    if inst_id == "punctuation:no_comma":
        return "," not in text

    if inst_id == "length_constraints:number_words":
        return _compare(_word_count(text), kwargs.get("relation"), kwargs.get("num_words"))

    if inst_id == "length_constraints:number_sentences":
        return _compare(_sentence_count(text), kwargs.get("relation"), kwargs.get("num_sentences"))

    if inst_id == "length_constraints:number_paragraphs":
        return _compare(len(_paragraphs(text)), kwargs.get("relation"), kwargs.get("num_paragraphs"))

    if inst_id == "length_constraints:nth_paragraph_first_word":
        paragraphs = _paragraphs(text)
        required_count = kwargs.get("num_paragraphs")
        if required_count is not None and len(paragraphs) != int(required_count):
            return False
        nth = int(kwargs.get("nth_paragraph", 1)) - 1
        if nth < 0 or nth >= len(paragraphs):
            return False
        first_word = kwargs.get("first_word", "").strip().lower()
        if not first_word:
            return False
        actual_words = WORD_RE.findall(paragraphs[nth])
        if not actual_words:
            return False
        return actual_words[0].lower() == first_word

    if inst_id == "keywords:forbidden_words":
        forbidden = [w.lower() for w in kwargs.get("forbidden_words") or [] if w]
        text_lower = text.lower()
        return not any(word in text_lower for word in forbidden)

    if inst_id == "keywords:existence":
        required = [w.lower() for w in kwargs.get("keywords") or [] if w]
        text_lower = text.lower()
        return all(word in text_lower for word in required)

    if inst_id == "keywords:frequency":
        return _compare(_count_keyword(text, kwargs.get("keyword", "")), kwargs.get("relation"), kwargs.get("frequency"))

    if inst_id == "keywords:letter_frequency":
        return _compare(_count_letter(text, kwargs.get("letter", "")), kwargs.get("let_relation"), kwargs.get("let_frequency"))

    if inst_id == "detectable_format:number_highlighted_sections":
        return _compare(_count_highlights(text), kwargs.get("relation") or "at least", kwargs.get("num_highlights"))

    if inst_id == "detectable_content:number_placeholders":
        return _compare(_count_placeholders(text), kwargs.get("relation") or "at least", kwargs.get("num_placeholders"))

    if inst_id == "detectable_format:number_bullet_lists":
        return _compare(_count_bullets(text), kwargs.get("relation") or "equal", kwargs.get("num_bullets"))

    if inst_id == "detectable_format:multiple_sections":
        splitter = (kwargs.get("section_spliter") or "").lower()
        if not splitter:
            return False
        count = text.lower().count(splitter.lower())
        return _compare(count, kwargs.get("relation") or "at least", kwargs.get("num_sections"))

    if inst_id == "detectable_format:title":
        return bool(re.search(r"<<[^<>]+>>", text))

    if inst_id == "detectable_format:json_format":
        sample = text.strip()
        if not (sample.startswith("{") or sample.startswith("[")):
            return False
        try:
            json.loads(sample)
            return True
        except json.JSONDecodeError:
            return False

    if inst_id == "detectable_format:constrained_response":
        return _check_constrained_response(text, prompt_text)

    if inst_id == "detectable_content:postscript":
        return _check_postscript(text, kwargs.get("postscript_marker"))

    if inst_id == "combination:repeat_prompt":
        expected = kwargs.get("prompt_to_repeat") or prompt_text
        if not expected:
            return False
        return text.strip().startswith(expected.strip())

    if inst_id == "combination:two_responses":
        if "******" in text:
            parts = [p for p in text.split("******") if p.strip()]
            return len(parts) >= 2
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return len(parts) >= 2

    if inst_id == "language:response_language":
        return _check_language(text, kwargs.get("language"))

    if inst_id == "startend:end_checker":
        phrase = kwargs.get("end_phrase")
        return bool(phrase and text.strip().endswith(phrase))

    if inst_id == "startend:quotation":
        trimmed = text.strip()
        return trimmed.startswith('"') and trimmed.endswith('"')

    if inst_id == "change_case:english_lowercase":
        letters = re.findall(r"[A-Za-z]", text)
        return all(ch.islower() for ch in letters)

    if inst_id == "change_case:english_capital":
        letters = re.findall(r"[A-Za-z]", text)
        return all(ch.isupper() for ch in letters) and bool(letters)

    if inst_id == "change_case:capital_word_frequency":
        return _compare(_count_uppercase_words(text), kwargs.get("capital_relation"), kwargs.get("capital_frequency"))

    if inst_id == "punctuation:no_period":
        return "." not in text

    if inst_id == "punctuation:no_colon":
        return ":" not in text

    if inst_id == "punctuation:no_semicolon":
        return ";" not in text

    return False


def evaluate_ifeval_instructions(output: str, instructions: List[Dict[str, Any]], prompt: Any = None) -> bool:
    if not instructions:
        return False
    response_text = extract_response_text(output)
    user_prompt = extract_user_prompt(prompt)
    for inst in instructions:
        if not evaluate_instruction(response_text, inst, user_prompt):
            return False
    return True
