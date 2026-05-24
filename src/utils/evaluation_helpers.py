import os
import re
from typing import Optional, Tuple

import dotenv

from .constants import (
    MEDIX_MODELS,
    MEDMO_MODELS,
    MEDVLTHINKER_MODELS,
    META_LLAMA_VISION_MODELS,
    QWEN_VL_MODELS,
)
from .prompt_builders import get_reordered_options

ANSWER_PATTERNS = [
    r"<answer>\s*([A-E])\s*</answer>",
    r"[Tt]he answer is\s*\[([A-E])\]",
    r"[Tt]he answer is\s*\(([A-E])\)",
    r"[Tt]he answer is\s*([A-E])\b",
    r"(?:[Aa]nswer|[Ss]hort answer):\s*\(([A-E])\)",
    r"(?:[Aa]nswer|[Ss]hort answer):\s*\[?([A-E])\]?",
    r"\*\*[Aa]nswer[:\*]*\s*\[?([A-E])\]?",
    r"(?:^|\n)\s*\(?([A-E])\)?\s*$",
    r"[Oo]ption\s+([A-E])\b",
    r"(?:choose|select|pick)\s+([A-E])\b",
]
NONE_PATTERNS = [
    r"[Nn]one of the above",
    r"[Nn]one of the (?:provided|given) options",
    r"correct answer is:?\s*(.+?)(?:\.|$)",
]
FREE_ANSWER_PATTERNS = [
    r"<answer>\s*(.+?)\s*</answer>",
    r"[Tt]he answer is:?\s*(.+?)(?:\.|$)",
    r"[Aa]nswer:\s*(.+?)(?:\.|$)",
    r"\*\*[Aa]nswer[:\*]*\s*(.+?)(?:\.|$)",
]

def _normalize_answer_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

def _collect_flamingo_prefix_segments(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    segments: list[str] = [stripped]
    first_line = stripped.splitlines()[0].strip()
    if first_line and first_line not in segments:
        segments.append(first_line)
    for separator in [".", ",", ";", ":", "(", "["]:
        prefix = first_line.split(separator, 1)[0].strip()
        if prefix and prefix not in segments:
            segments.append(prefix)
    first_tokens = " ".join(first_line.split()[:3]).strip()
    if first_tokens and first_tokens not in segments:
        segments.append(first_tokens)
    return segments

def _match_option_text(
    model_output: str, options: dict[str, str]
) -> Tuple[Optional[str], Optional[str]]:
    normalized_output = _normalize_answer_text(model_output)
    if not normalized_output:
        return None, None
    for label, option_text in sorted(options.items()):
        normalized_option = _normalize_answer_text(option_text)
        if not normalized_option:
            continue
        if normalized_output == normalized_option:
            return label, option_text
        if normalized_output.startswith(normalized_option):
            return label, option_text
    return None, None

def _extract_llava_pred(
    model_output: str,
    options: dict[str, str],
) -> Tuple[Optional[str], Optional[str]]:
    if not model_output:
        return None, None
    text = model_output.strip()
    option_items = sorted(options.items())
    option_labels = [label for label, _ in option_items]
    if not option_labels:
        return None, None
    matched_label, matched_text = _match_option_text(text, options)
    if matched_label:
        return matched_label, matched_text
    letter_pattern = r"(?:^|\n|[([])\s*({})(?=\s|$|[.)\]:<])".format(
        "|".join(re.escape(label) for label in option_labels)
    )
    candidates = re.findall(letter_pattern, text)
    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) > 1:
        parsed = [candidate for candidate in candidates if candidate in option_labels]
        if len(set(parsed)) == 1:
            return parsed[0], None
        return None, None
    normalized_output = _normalize_answer_text(text)
    matched: list[tuple[str, str]] = []
    for label, option_text in option_items:
        normalized_option = _normalize_answer_text(option_text)
        if not normalized_option:
            continue
        pattern = r"(?:^|\s){}(?=\s|$|\.|</s>)".format(re.escape(normalized_option))
        if re.search(pattern, normalized_output):
            matched.append((label, option_text))
    if len(matched) == 1:
        return matched[0]
    return None, None

def _extract_flamingo_pred(
    model_output: str,
    options: dict[str, str],
) -> Tuple[Optional[str], Optional[str]]:
    if not model_output:
        return None, None
    text = model_output.strip()
    option_items = sorted(options.items())
    option_labels = [label for label, _ in option_items]
    if not option_labels:
        return None, None
    for segment in _collect_flamingo_prefix_segments(text):
        matched_label, matched_text = _match_option_text(segment, options)
        if matched_label:
            return matched_label, matched_text
    letter_pattern = r"(?:^|\n|[([])\s*({})(?=\s|$|[.)\]:<])".format(
        "|".join(re.escape(label) for label in option_labels)
    )
    candidates = re.findall(letter_pattern, text)
    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) > 1:
        parsed = [candidate for candidate in candidates if candidate in option_labels]
        if len(set(parsed)) == 1:
            return parsed[0], None
        return None, None
    normalized_output = _normalize_answer_text(text)
    matched: list[tuple[str, str]] = []
    for label, option_text in option_items:
        normalized_option = _normalize_answer_text(option_text)
        if not normalized_option:
            continue
        pattern = r"(?:^|\s){}(?=\s|$|\.|</s>)".format(re.escape(normalized_option))
        if re.search(pattern, normalized_output):
            matched.append((label, option_text))
    if len(matched) == 1:
        return matched[0]
    return None, None

def _extract_medvlthinker_pred(
    model_output: str,
    options: dict[str, str],
    options_mode: str,
) -> Tuple[Optional[str], Optional[str]]:
    if not model_output:
        return None, None
    text = model_output.strip()
    answer_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL
    )
    if not answer_match:
        return None, None
    answer_text = answer_match.group(1).strip()
    if not answer_text:
        return None, None
    if options_mode == "no_options":
        return None, answer_text
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        if re.search(r"\bnone of the above\b", answer_text, re.IGNORECASE):
            return "NONE", answer_text
    label_match = re.match(
        r"^\(?\s*([A-E])\s*[\)\].:\-]?\s*(.*)$", answer_text, re.DOTALL
    )
    if label_match:
        label = label_match.group(1).upper()
        trailing_text = label_match.group(2).strip() or None
        return label, trailing_text
    matched_label, matched_text = _match_option_text(answer_text, options)
    if matched_label:
        return matched_label, matched_text
    return None, answer_text

def _extract_qwen25_vl_pred(
    model_output: str,
    options: dict[str, str],
    options_mode: str,
) -> Tuple[Optional[str], Optional[str]]:
    if not model_output:
        return None, None
    text = model_output.strip()
    first_line = text.splitlines()[0].strip() if text.splitlines() else text
    if options_mode == "no_options":
        return None, text
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        if re.search(r"^\s*none of the above\s*\.?\s*$", text, re.IGNORECASE):
            return "NONE", "None of the above"
    for candidate_text in (first_line, text):
        prefixed_match = re.match(
            r"^\s*(?:(?i:answer|correct answer))\s*[:\-]?\s*([A-D])\s*[\.\)\]:-]?\s*(.*?)\s*$",
            candidate_text,
        )
        if prefixed_match:
            label = prefixed_match.group(1).upper()
            trailing_text = prefixed_match.group(2).strip() or None
            return label, trailing_text
        punct_match = re.match(
            r"^\s*\(?([A-D])\)?\s*[\.\)\]:-]\s*(.*?)\s*$",
            candidate_text,
        )
        if punct_match:
            label = punct_match.group(1).upper()
            trailing_text = punct_match.group(2).strip() or None
            return label, trailing_text
        spaced_match = re.match(r"^\s*\(?([A-D])\)?\s+(.+?)\s*$", candidate_text)
        if spaced_match:
            label = spaced_match.group(1).upper()
            trailing_text = spaced_match.group(2).strip()
            normalized_option = _normalize_answer_text(options.get(label, ""))
            if normalized_option and _normalize_answer_text(trailing_text).startswith(
                normalized_option
            ):
                return label, options[label]
        bare_match = re.match(r"^\s*\(?([A-D])\)?\s*$", candidate_text)
        if bare_match:
            return bare_match.group(1).upper(), None
    line_match = re.search(
        r"(?m)^\s*(?:(?i:answer|correct answer))\s*[:\-]?\s*([A-D])\s*[\.\)\]:-]?\s*(.*?)\s*$",
        text,
    )
    if line_match:
        label = line_match.group(1).upper()
        trailing_text = line_match.group(2).strip() or None
        return label, trailing_text
    for candidate_text in (first_line, text):
        matched_label, matched_text = _match_option_text(candidate_text, options)
        if matched_label:
            return matched_label, matched_text
    return None, text

def _extract_adapt_qwen2_pred(
    model_output: str,
    options: dict[str, str],
    options_mode: str,
) -> Tuple[Optional[str], Optional[str]]:
    if not model_output:
        return None, None
    text = model_output.strip()
    first_line = text.splitlines()[0].strip() if text.splitlines() else text
    if options_mode == "no_options":
        return None, text
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        if re.search(r"\bnone of the above\b", text, re.IGNORECASE):
            return "NONE", "None of the above"
    for candidate_text in (first_line, text):
        prefixed_match = re.match(
            r"^\s*(?:(?i:answer|correct answer))\s*[:\-]?\s*([A-D])\s*[\.\)\]:-]?\s*(.*?)\s*$",
            candidate_text,
        )
        if prefixed_match:
            label = prefixed_match.group(1).upper()
            trailing_text = prefixed_match.group(2).strip() or None
            return label, trailing_text
        punct_match = re.match(r"^\s*([A-D])\s*[\.\)\]:-]\s*(.*?)\s*$", candidate_text)
        if punct_match:
            label = punct_match.group(1).upper()
            trailing_text = punct_match.group(2).strip() or None
            return label, trailing_text
        spaced_match = re.match(r"^\s*([A-D])\s+(.+?)\s*$", candidate_text)
        if spaced_match:
            label = spaced_match.group(1).upper()
            trailing_text = spaced_match.group(2).strip()
            normalized_option = _normalize_answer_text(options.get(label, ""))
            if normalized_option and _normalize_answer_text(trailing_text).startswith(
                normalized_option
            ):
                return label, options[label]
        bare_match = re.match(r"^\s*([A-D])\s*$", candidate_text)
        if bare_match:
            return bare_match.group(1).upper(), None
    answer_text_patterns = [
        r"(?is)(?:therefore,\s*)?(?:the\s+correct\s+answer\s+to\s+the\s+question\s+is|the\s+correct\s+answer\s+is|the\s+answer\s+to\s+the\s+question\s+is|the\s+answer\s+is)\s*[:\-]?\s*[\[\('\"]*\s*([a-z][a-z0-9\s\-_/]+?)\s*[\]'\")]*\s*$",
        r"(?is)(?:therefore,\s*)?(?:the\s+correct\s+answer\s+to\s+the\s+question\s+is|the\s+correct\s+answer\s+is|the\s+answer\s+to\s+the\s+question\s+is|the\s+answer\s+is)\s*[:\-]?\s*[\[\('\"]*\s*([a-z][a-z0-9\s\-_/]+?)\s*[\]'\")]*\s*[\.,]?\s*$",
    ]
    for pattern in answer_text_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        answer_text = match.group(1).strip()
        matched_label, matched_text = _match_option_text(answer_text, options)
        if matched_label:
            return matched_label, matched_text
        return None, answer_text
    option_items = sorted(options.items())
    normalized_text = _normalize_answer_text(text)
    matched_suffixes: list[tuple[str, str]] = []
    for label, option_text in option_items:
        normalized_option = _normalize_answer_text(option_text)
        if not normalized_option:
            continue
        if normalized_text.endswith(normalized_option):
            matched_suffixes.append((label, option_text))
    if len(matched_suffixes) == 1:
        return matched_suffixes[0]
    for candidate_text in (first_line, text):
        matched_label, matched_text = _match_option_text(candidate_text, options)
        if matched_label:
            return matched_label, matched_text
    return None, text

def _extract_gemma_medgemma_pred(
    model_output: str,
    options: dict[str, str],
    options_mode: str,
) -> Tuple[Optional[str], Optional[str]]:
    if not model_output:
        return None, None
    text = model_output.strip()
    cleaned_text = re.sub(r"[*_`]+", "", text)
    first_line = (
        cleaned_text.splitlines()[0].strip()
        if cleaned_text.splitlines()
        else cleaned_text
    )
    if options_mode == "no_options":
        return None, cleaned_text
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        if re.search(r"\bnone of the above\b", cleaned_text, re.IGNORECASE):
            return "NONE", "None of the above"
    option_labels = [label for label, _ in sorted(options.items())]
    if not option_labels:
        return None, None
    label_pattern = "|".join(re.escape(label) for label in option_labels)
    prefixed_patterns = [
        rf"(?im)(?:therefore,\s*)?(?:the\s+correct\s+answer(?:\s+to\s+the\s+question)?\s+is|the\s+answer(?:\s+to\s+the\s+question)?\s+is)\s*[:\-]?\s*[\[\(]?\s*({label_pattern})\s*[\]\)]?\s*[\.\)\]:-]?\s*([^\n]*)",
        rf"(?im)^\s*(?:answer|correct answer)\s*[:\-]?\s*[\[\(]?\s*({label_pattern})\s*[\]\)]?\s*[\.\)\]:-]?\s*([^\n]*)",
    ]
    for pattern in prefixed_patterns:
        match = re.search(pattern, cleaned_text)
        if match:
            return match.group(1).upper(), match.group(2).strip() or None
    line_patterns = [
        rf"(?im)^\s*({label_pattern})\s*[\.\)\]:-]\s*(.*?)\s*$",
        rf"(?im)^\s*\(?({label_pattern})\)?\s*$",
    ]
    for pattern in line_patterns:
        match = re.search(pattern, cleaned_text)
        if match:
            trailing = (
                match.group(2).strip()
                if match.lastindex and match.lastindex >= 2
                else None
            )
            return match.group(1).upper(), trailing or None
    inline_label_match = re.search(
        rf"(?m)(?:^|[\s,;:(\[])({label_pattern})\s*[\.\)\]:-]\s*([^\n]*)",
        cleaned_text,
    )
    if inline_label_match:
        return (
            inline_label_match.group(1).upper(),
            inline_label_match.group(2).strip() or None,
        )
    for candidate_text in (first_line, cleaned_text):
        matched_label, matched_text = _match_option_text(candidate_text, options)
        if matched_label:
            return matched_label, matched_text
    return None, cleaned_text

def _extract_qwen2_vl_pred(
    model_output: str,
    options: dict[str, str],
    options_mode: str,
) -> Tuple[Optional[str], Optional[str]]:
    return _extract_qwen25_vl_pred(model_output, options, options_mode)

def _extract_adapt_internvl3_pred(
    model_output: str,
    options: dict[str, str],
    options_mode: str,
) -> Tuple[Optional[str], Optional[str]]:
    if not model_output:
        return None, None
    text = model_output.strip()
    first_line = text.splitlines()[0].strip() if text.splitlines() else text
    last_line = text.splitlines()[-1].strip() if text.splitlines() else text
    if options_mode == "no_options":
        return None, text
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        if re.search(r"\bnone of the above\b", text, re.IGNORECASE):
            return "NONE", "None of the above"
    for candidate_text in (first_line, text):
        prefixed_match = re.match(
            r"^\s*(?:(?i:answer|correct answer))\s*[:\-]?\s*([A-D])\s*[\.\)\]:-]?\s*(.*?)\s*$",
            candidate_text,
        )
        if prefixed_match:
            label = prefixed_match.group(1).upper()
            trailing_text = prefixed_match.group(2).strip() or None
            return label, trailing_text
        punct_match = re.match(r"^\s*([A-D])\s*[\.\)\]:-]\s*(.*?)\s*$", candidate_text)
        if punct_match:
            label = punct_match.group(1).upper()
            trailing_text = punct_match.group(2).strip() or None
            return label, trailing_text
        spaced_match = re.match(r"^\s*([A-D])\s+(.+?)\s*$", candidate_text)
        if spaced_match:
            label = spaced_match.group(1).upper()
            trailing_text = spaced_match.group(2).strip()
            normalized_option = _normalize_answer_text(options.get(label, ""))
            if normalized_option and _normalize_answer_text(trailing_text).startswith(
                normalized_option
            ):
                return label, options[label]
        bare_match = re.match(r"^\s*([A-D])\s*$", candidate_text)
        if bare_match:
            return bare_match.group(1).upper(), None
    answer_text_patterns = [
        r"(?i)the answer(?:\s+to\s+(?:the\s+)?question)?\s+is[:\s]+([^\n\.,\(\)]+?)(?:[,\.]|[\.\s]*$)",
        r"(?s).*\n\n(yes|no)\s*$",
        r"(?i)(?:(?:does|did)\s+(?:not\s+)?contain|(?:picture|image)\s+contains?)\s+(?:a\s+|the\s+|lung\s+|brain\s+|kidney\s+)?([a-z][\w\s]+?)(?:\.|,|$)",
        r"(?i)(?:largest|smallest|bigger|smaller|larger)\s+organ\s+in\s+this\s+image\s+is\s+(?:the\s+)?([a-z][\w\s]+?)[\.\s]*$",
        r"(?i)this is (?:indeed )?a study of (?:[a-z\s]+,\s*specifically\s+the\s+)?([a-z][\w\s]+?)[\.\s]*$",
        r"(?i)\bis\s+(?:the\s+)?([a-z][\w\s]+?)\s*[\.!]?\s*$",
        r"(?i)(?:^|\n)(yes|no|t[12]|left|right|hyperdense|hypodense|heart|lung|liver|spleen|kidney|colon|rectum|bladder|stomach|pancreas|small bowel|brain)\s*$",
    ]
    extracted_answer_text = None
    for candidate_text in (text, last_line):
        for pattern in answer_text_patterns:
            match = re.search(pattern, candidate_text, re.IGNORECASE)
            if not match:
                continue
            answer_text = match.group(1).strip()
            matched_label, matched_text = _match_option_text(answer_text, options)
            if matched_label:
                return matched_label, matched_text
            extracted_answer_text = answer_text
            break
        if extracted_answer_text:
            break
    normalized_option_map = {
        label: _normalize_answer_text(option_text)
        for label, option_text in options.items()
    }
    yes_label = next(
        (label for label, option in normalized_option_map.items() if option == "yes"),
        None,
    )
    no_label = next(
        (label for label, option in normalized_option_map.items() if option == "no"),
        None,
    )
    if yes_label and no_label:
        affirmative_patterns = [
            r"(?i)\b(?:is|are|was|were)\s+(?:indeed\s+)?(?:visualized|visible|seen|present|identified|noted)\b",
            r"(?i)\bthere\s+(?:is|are)\s+(?:clear\s+)?(?:evidence|presence)\b",
            r"(?i)\b(?:appears?|appear)\s+to\s+be\s+(?:present|visible|intact|normal)\b",
        ]
        negative_patterns = [
            r"(?i)\b(?:is|are|was|were)\s+not\s+(?:visualized|visible|seen|present|identified)\b",
            r"(?i)\bno\s+(?:clear\s+)?(?:evidence|sign|signs|indication|indications|presence)\b",
            r"(?i)\bwithout\s+(?:any\s+)?(?:visible\s+)?(?:evidence|sign|signs|indication|indications)\b",
            r"(?i)\babsence\s+of\b",
        ]
        for pattern in negative_patterns:
            if re.search(pattern, text):
                return no_label, options[no_label]
        for pattern in affirmative_patterns:
            if re.search(pattern, text):
                return yes_label, options[yes_label]
    line_label_match = re.search(
        r"(?m)^\s*(?:(?i:answer|correct answer))\s*[:\-]?\s*([A-D])\s*[\.\)\]:-]?\s*(.*?)\s*$",
        text,
    )
    if line_label_match:
        label = line_label_match.group(1).upper()
        trailing_text = line_label_match.group(2).strip() or None
        return label, trailing_text
    if extracted_answer_text is not None:
        return None, extracted_answer_text
    option_items = sorted(options.items())
    normalized_text = _normalize_answer_text(text)
    matched_suffixes: list[tuple[str, str]] = []
    for label, option_text in option_items:
        normalized_option = _normalize_answer_text(option_text)
        if not normalized_option:
            continue
        if normalized_text.endswith(normalized_option):
            matched_suffixes.append((label, option_text))
    if len(matched_suffixes) == 1:
        return matched_suffixes[0]
    for candidate_text in (first_line, last_line, text):
        matched_label, matched_text = _match_option_text(candidate_text, options)
        if matched_label:
            return matched_label, matched_text
    return None, text

def _extract_medix_pred(
    model_output: str,
    options: dict[str, str],
    options_mode: str,
) -> Tuple[Optional[str], Optional[str]]:
    if not model_output:
        return None, None
    text = model_output.strip()
    answer_matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL
    )
    if not answer_matches:
        return None, None
    answer_text = answer_matches[-1].strip()
    if not answer_text:
        return None, None
    if options_mode == "no_options":
        return None, answer_text
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        if re.search(r"\bnone of the above\b", answer_text, re.IGNORECASE):
            return "NONE", answer_text
    label_patterns = [
        r"\b(?:(?i:the\s+correct\s+answer|correct\s+answer|the\s+answer|answer))\s+is\s+([A-E])\b",
        r"\b(?:(?i:option))\s+([A-E])\b",
        r"^\s*\(?([A-E])\)?\s*[\].:\-]?\s*(.*)$",
    ]
    for pattern in label_patterns:
        match = re.search(pattern, answer_text, re.DOTALL)
        if not match:
            continue
        label = match.group(1).upper()
        trailing_text = None
        if match.lastindex and match.lastindex >= 2:
            trailing_text = match.group(2).strip() or None
        return label, trailing_text
    matched_label, matched_text = _match_option_text(answer_text, options)
    if matched_label:
        return matched_label, matched_text
    return None, answer_text

def _extract_adapt_llama_pred(
    model_output: str,
    options: dict[str, str],
    options_mode: str,
) -> Tuple[Optional[str], Optional[str]]:
    if not model_output:
        return None, None
    text = model_output.strip()
    if options_mode == "no_options":
        answer_patterns = [
            r"(?is)(?:the\s+answer\s+to\s+the\s+question\s+is|the\s+answer\s+is)\s*[:\-]?\s*[\[\('\"]*\s*([a-z][a-z\s\-]+?)\s*[\]'\")]*\s*$",
            r"(?is)[\[\('\"]+\s*([a-z][a-z\s\-]+?)\s*[\]'\")]+\s*$",
            r"(?is)\b(yes|no)\b\s*$",
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, text)
            if match:
                return None, match.group(1).strip()
        return None, text
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        if re.search(r"\bnone of the above\b", text, re.IGNORECASE):
            return "NONE", "None of the above"
    label_patterns = [
        r"\b(?:(?i:the\s+answer\s+is|the\s+correct\s+answer\s+is|answer\s*[:\-]?))\s*\[?([A-E])\]?\b",
        r"\b(?:(?i:option))\s+([A-E])\b",
    ]
    for pattern in label_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper(), None
    answer_text_patterns = [
        r"(?is)(?:the\s+answer\s+to\s+the\s+question\s+is|the\s+answer\s+is|answer\s*[:\-]?)\s*[\[\('\"]*\s*([a-z][a-z\s\-]+?)\s*[\]'\")]*\s*$",
        r"(?is)[\[\('\"]+\s*([a-z][a-z\s\-]+?)\s*[\]'\")]+\s*$",
        r"(?is)\b(yes|no)\b\s*$",
    ]
    for pattern in answer_text_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        answer_text = match.group(1).strip()
        matched_label, matched_text = _match_option_text(answer_text, options)
        if matched_label:
            return matched_label, matched_text
        return None, answer_text
    matched_label, matched_text = _match_option_text(text, options)
    if matched_label:
        return matched_label, matched_text
    return None, None

def _extract_meta_llama32_vision_pred(
    model_output: str,
    options: dict[str, str],
    options_mode: str,
) -> Tuple[Optional[str], Optional[str]]:
    if not model_output:
        return None, None
    text = model_output.strip()
    first_line = text.splitlines()[0].strip() if text.splitlines() else text
    if options_mode == "no_options":
        return None, text
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        if re.search(r"^\s*none of the above\s*\.?\s*$", text, re.IGNORECASE):
            return "NONE", "None of the above"
    label_patterns = [
        r"^\s*([A-D])\s*[\.\)\]:-]?\s*$",
        r"(?im)^\s*([A-D])\s*[\)\].:\-]\s+.+?\s*$",
        r"(?im)^\s*\*\*[Aa]nswer[:\*]*\s*([A-D])\s*[\.\)\]]?\s*$",
        r"(?im)^\s*(?:(?i:answer|final answer|correct answer))\s*[:\-]?\s*([A-D])\s*[\.\)\]]?\s*$",
        r"(?im)^\s*(?:(?i:answer|final answer|correct answer))\s*[:\-]?\s*([A-D])\s*[\)\].:\-]\s+.+?\s*$",
        r"\b(?:(?i:answer|final answer|correct answer))\s*[:\-]?\s*([A-D])\b",
        r"\b(?:(?i:option))\s+([A-D])\b",
    ]
    for candidate in (text, first_line):
        for pattern in label_patterns:
            match = re.search(pattern, candidate)
            if match:
                return match.group(1).upper(), None
    matched_label, matched_text = _match_option_text(first_line, options)
    if matched_label:
        return matched_label, matched_text
    matched_label, matched_text = _match_option_text(text, options)
    if matched_label:
        return matched_label, matched_text
    return None, None

def extract_answer_label(
    model_output: str,
    options_mode: str = "with_options",
) -> Tuple[Optional[str], Optional[str]]:
    if not model_output:
        return None, None
    text = model_output.strip()
    if options_mode == "no_options":
        for pattern in FREE_ANSWER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return None, match.group(1).strip()
        return None, text
    for pattern in ANSWER_PATTERNS:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(1).upper(), None
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        for pattern in NONE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                correct_answer = match.group(1).strip() if match.lastindex else None
                return "NONE", correct_answer
    return None, None

def gpt4o_judge(
    question: str,
    options: dict[str, str],
    correct_answer: str,
    correct_label: str,
    model_output: str,
    options_mode: str = "with_options",
) -> Tuple[bool, str]:
    dotenv.load_dotenv("./.env")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return False, "No OpenAI API key found for GPT-4O judge"
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    if options_mode == "no_options":
        judge_prompt = f"""You are a medical AI evaluation judge. Determine if the model's response is semantically equivalent to the correct answer.
Question: {question}
Correct Answer: {correct_answer}
Model's Response: {model_output}
Is the model's response correct? Consider semantic equivalence - the model doesn't need to use the exact same words, but the meaning must match.
Respond with EXACTLY one line:
CORRECT - if the model's answer matches the correct answer
INCORRECT - if the model's answer does not match"""
    else:
        options_str = "\n".join(f"  {k}) {v}" for k, v in sorted(options.items()))
        judge_prompt = f"""You are a medical AI evaluation judge. Determine which option (if any) the model selected.
Question: {question}
Options:
{options_str}
Correct Answer: [{correct_label}] {correct_answer}
Model's Response: {model_output}
Based on the model's response, did it select the correct answer [{correct_label}]?
Respond with EXACTLY one line:
CORRECT - if the model selected option {correct_label}
INCORRECT - if the model selected a different option or gave no clear answer"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a precise evaluation judge."},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
            max_tokens=50,
        )
        judge_output = response.choices[0].message.content.strip()
        is_correct = judge_output.upper().startswith("CORRECT")
        return is_correct, judge_output
    except Exception as e:
        return False, f"GPT-4O judge error: {str(e)}"

def evaluate_single(
    model_output: str,
    sample,
    options_mode: str = "with_options",
    options_order: str = "default",
    use_gpt_judge: bool = True,
    model_stem: Optional[str] = None,
) -> dict:
    effective_options, effective_answer_label = get_reordered_options(
        sample,
        options_order=options_order,
    )
    result = {
        "idx": sample.idx,
        "gt_label": effective_answer_label,
        "gt_answer": sample.answer,
        "model_output": model_output,
        "options_mode": options_mode,
        "options_order": options_order,
    }
    use_llava_extractor = model_stem in {"llava-med-7b", "llava-v0-7b"}
    use_flamingo_extractor = model_stem in {
        "med-flamingo-9b",
        "random_med-flamingo-9b_1",
        "random_med-flamingo-9b_2",
        "random_med-flamingo-9b_3",
        "open-flamingo-9b",
        "random_open-flamingo-9b_1",
        "random_open-flamingo-9b_2",
        "random_open-flamingo-9b_3",
    }
    use_medvlthinker_extractor = model_stem in MEDVLTHINKER_MODELS
    use_medix_extractor = model_stem in MEDIX_MODELS
    use_adapt_llama_extractor = model_stem == "adapt-llama3.2-11b"
    use_adapt_qwen2_extractor = model_stem == "adapt-qwen2-2b"
    use_qwen2_vl_extractor = model_stem == "qwen2-vl-2b-instruct"
    use_adapt_internvl3_extractor = model_stem in {
        "adapt-internVL3-1b",
        "internvl3-1b",
        "random_InternVL3-1B_1",
        "random_InternVL3-1B_2",
        "random_InternVL3-1B_3",
    }
    use_meta_llama32_vision_extractor = model_stem in META_LLAMA_VISION_MODELS
    use_gemma_medgemma_extractor = (
        model_stem in {"gemma", "gemma-27b", "medgemma", "medgemma-27b"}
        or model_stem.startswith("random_gemma_4b_")
        or model_stem.startswith("random_gemma_27b_")
        or model_stem.startswith("random_medgemma_4b_")
        or model_stem.startswith("random_medgemma_27b_")
    )
    use_qwen25_vl_extractor = model_stem in MEDMO_MODELS or model_stem in QWEN_VL_MODELS
    if use_medvlthinker_extractor:
        pred_label, pred_text = _extract_medvlthinker_pred(
            model_output,
            effective_options,
            options_mode,
        )
    elif use_medix_extractor:
        pred_label, pred_text = _extract_medix_pred(
            model_output,
            effective_options,
            options_mode,
        )
    elif use_adapt_llama_extractor:
        pred_label, pred_text = _extract_adapt_llama_pred(
            model_output,
            effective_options,
            options_mode,
        )
    elif use_adapt_qwen2_extractor:
        pred_label, pred_text = _extract_adapt_qwen2_pred(
            model_output,
            effective_options,
            options_mode,
        )
    elif use_qwen2_vl_extractor:
        pred_label, pred_text = _extract_qwen2_vl_pred(
            model_output,
            effective_options,
            options_mode,
        )
    elif use_adapt_internvl3_extractor:
        pred_label, pred_text = _extract_adapt_internvl3_pred(
            model_output,
            effective_options,
            options_mode,
        )
    elif use_meta_llama32_vision_extractor:
        pred_label, pred_text = _extract_meta_llama32_vision_pred(
            model_output,
            effective_options,
            options_mode,
        )
    elif use_gemma_medgemma_extractor:
        pred_label, pred_text = _extract_gemma_medgemma_pred(
            model_output,
            effective_options,
            options_mode,
        )
    elif use_qwen25_vl_extractor:
        pred_label, pred_text = _extract_qwen25_vl_pred(
            model_output,
            effective_options,
            options_mode,
        )
    else:
        pred_label, pred_text = extract_answer_label(model_output, options_mode)
    if options_mode == "no_options":
        if pred_text and pred_text.lower().strip() == sample.answer.lower().strip():
            result["correct"] = 1
            result["pred_label"] = effective_answer_label
            result["pred_text"] = pred_text
            result["method"] = "regex_exact"
        elif use_gpt_judge:
            is_correct, explanation = gpt4o_judge(
                sample.question,
                effective_options,
                sample.answer,
                effective_answer_label,
                model_output,
                options_mode,
            )
            result["correct"] = int(is_correct)
            result["pred_label"] = effective_answer_label if is_correct else "UNKNOWN"
            result["pred_text"] = pred_text or model_output[:200]
            result["method"] = "gpt_judge"
            result["judge_explanation"] = explanation
        else:
            result["correct"] = 0
            result["pred_label"] = "UNKNOWN"
            result["pred_text"] = pred_text or model_output[:200]
            result["method"] = "regex_failed"
    elif options_mode in ("incorrect_options", "incorrect_options_blind"):
        if pred_label == "NONE":
            result["correct"] = 1
            result["pred_label"] = "NONE"
            result["pred_text"] = pred_text
            result["method"] = "regex_none_detected"
        elif (
            use_medix_extractor
            and pred_text
            and _normalize_answer_text(pred_text)
            == _normalize_answer_text(sample.answer)
        ):
            result["correct"] = 1
            result["pred_label"] = "NONE"
            result["pred_text"] = pred_text
            result["method"] = "regex_answer_text"
        elif pred_label:
            result["correct"] = 0
            result["pred_label"] = pred_label
            result["pred_text"] = None
            result["method"] = "regex"
        elif _normalize_answer_text(model_output) == _normalize_answer_text(
            sample.answer
        ):
            result["correct"] = 1
            result["pred_label"] = "NONE"
            result["pred_text"] = sample.answer
            result["method"] = "exact_answer_text"
        elif use_gpt_judge:
            is_correct, explanation = gpt4o_judge(
                sample.question,
                effective_options,
                sample.answer,
                effective_answer_label,
                model_output,
                options_mode,
            )
            result["correct"] = int(is_correct)
            result["pred_label"] = "NONE" if is_correct else "UNKNOWN"
            result["pred_text"] = model_output[:200]
            result["method"] = "gpt_judge"
            result["judge_explanation"] = explanation
        else:
            result["correct"] = 0
            result["pred_label"] = "UNKNOWN"
            result["method"] = "regex_failed"
    else:
        if pred_label:
            result["correct"] = int(pred_label == effective_answer_label)
            result["pred_label"] = pred_label
            if use_medvlthinker_extractor:
                result["method"] = "medvlthinker_extract"
            elif use_adapt_llama_extractor:
                result["method"] = "adapt_llama_extract"
            elif use_adapt_qwen2_extractor:
                result["method"] = "adapt_qwen2_extract"
            elif use_meta_llama32_vision_extractor:
                result["method"] = "meta_llama32_vision_extract"
            elif use_gemma_medgemma_extractor:
                result["method"] = "gemma_medgemma_extract"
            elif use_qwen25_vl_extractor:
                result["method"] = "qwen25_vl_extract"
            else:
                result["method"] = "regex"
        else:
            if use_medvlthinker_extractor:
                matched_label, matched_text = _match_option_text(
                    pred_text or model_output, effective_options
                )
                method = "medvlthinker_option_text_match"
            elif use_qwen25_vl_extractor:
                matched_label, matched_text = _match_option_text(
                    pred_text or model_output, effective_options
                )
                method = "qwen25_vl_option_text_match"
            elif use_adapt_qwen2_extractor:
                matched_label, matched_text = _match_option_text(
                    pred_text or model_output, effective_options
                )
                method = "adapt_qwen2_option_text_match"
            elif use_adapt_llama_extractor:
                matched_label, matched_text = _match_option_text(
                    pred_text or model_output, effective_options
                )
                method = "adapt_llama_option_text_match"
            elif use_meta_llama32_vision_extractor:
                matched_label, matched_text = _match_option_text(
                    pred_text or model_output, effective_options
                )
                method = "meta_llama32_vision_option_text_match"
            elif use_gemma_medgemma_extractor:
                matched_label, matched_text = _match_option_text(
                    pred_text or model_output, effective_options
                )
                method = "gemma_medgemma_option_text_match"
            elif use_llava_extractor:
                matched_label, matched_text = _extract_llava_pred(
                    model_output, effective_options
                )
                method = "llava_option_extract"
            elif use_flamingo_extractor:
                matched_label, matched_text = _extract_flamingo_pred(
                    model_output, effective_options
                )
                method = "flamingo_option_extract"
            else:
                matched_label, matched_text = _match_option_text(
                    model_output, effective_options
                )
                method = "option_text_match"
            if matched_label:
                result["correct"] = int(matched_label == effective_answer_label)
                result["pred_label"] = matched_label
                result["pred_text"] = matched_text
                result["method"] = method
            elif use_gpt_judge:
                is_correct, explanation = gpt4o_judge(
                    sample.question,
                    effective_options,
                    sample.answer,
                    effective_answer_label,
                    model_output,
                    options_mode,
                )
                result["correct"] = int(is_correct)
                result["pred_label"] = (
                    effective_answer_label if is_correct else "UNKNOWN"
                )
                result["method"] = "gpt_judge"
                result["judge_explanation"] = explanation
            else:
                result["correct"] = 0
                result["pred_label"] = "UNKNOWN"
                result["method"] = "regex_failed"
    return result

def compute_retrieval_failures(
    probe_predictions: list[dict],
    generation_predictions: list[dict],
    ground_truth_labels: dict[int, str],
) -> dict:
    probe_map = {r["idx"]: r.get("pred_label", "") for r in probe_predictions}
    gen_map = {}
    for r in generation_predictions:
        idx = r["idx"]
        if "correct" in r:
            gen_map[idx] = bool(r["correct"])
        else:
            gen_map[idx] = r.get("pred_label", "") == ground_truth_labels.get(idx, "")
    results = {
        "retrieval_failure": [],
        "both_correct": [],
        "both_wrong": [],
        "gen_only_correct": [],
    }
    all_idxs = set(probe_map.keys()) & set(gen_map.keys())
    for idx in sorted(all_idxs):
        gt = ground_truth_labels.get(idx, "")
        probe_correct = probe_map[idx] == gt
        gen_correct = gen_map[idx] if isinstance(gen_map[idx], bool) else gen_map[idx]
        if probe_correct and gen_correct:
            results["both_correct"].append(idx)
        elif probe_correct and not gen_correct:
            results["retrieval_failure"].append(idx)
        elif not probe_correct and gen_correct:
            results["gen_only_correct"].append(idx)
        else:
            results["both_wrong"].append(idx)
    total = max(len(all_idxs), 1)
    results["rates"] = {
        "retrieval_failure_rate": len(results["retrieval_failure"]) / total,
        "both_correct_rate": len(results["both_correct"]) / total,
        "both_wrong_rate": len(results["both_wrong"]) / total,
        "gen_only_correct_rate": len(results["gen_only_correct"]) / total,
        "total_samples": len(all_idxs),
    }
    return results
