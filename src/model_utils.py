"""
GrammarAI inference utilities.

Selected grammar-correction baseline:
    Checkpoint: vennify/t5-base-grammar-correction
    Framework: PyTorch
    Default device: CPU
    Paragraph segmentation: PySBD

Word highlighting is derived by aligning the original text with the corrected
text. It identifies model changes, not independently verified grammar errors.
No training or fine-tuning is performed by this module.
"""
from __future__ import annotations

import html
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Union

import torch
import pysbd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL_NAME = "vennify/t5-base-grammar-correction"
DEFAULT_MAX_INPUT_TOKENS = 512
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_NUM_BEAMS = 5

DeviceLike = Union[str, torch.device]

# Word tokens retain contractions and apostrophes as single units.
_WORD_PATTERN = re.compile(r"\b[\w]+(?:['’][\w]+)*\b", flags=re.UNICODE)


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: DeviceLike = "cpu",
    local_files_only: bool = False,
):
    """Load the tokenizer and model once; return (tokenizer, model, device)."""
    resolved_device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model.to(resolved_device)
    model.eval()

    return tokenizer, model, resolved_device


def _model_device(model: torch.nn.Module) -> torch.device:
    """Return the device holding the model parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("The supplied model has no parameters.") from exc


def _capitalize_sentence_start(text: str) -> str:
    """Capitalize the first alphabetic character; formatting safeguard only."""
    match = re.search(r"[A-Za-z]", text)
    if match is None:
        return text
    index = match.start()
    return text[:index] + text[index].upper() + text[index + 1:]


def correct_grammar(
    text: str,
    tokenizer,
    model: torch.nn.Module,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    num_beams: int = DEFAULT_NUM_BEAMS,
    capitalize_initial: bool = False,
) -> str:
    """
    Correct one sentence with the Vennify T5 checkpoint.

    Uses the tested 'grammar: ' prefix and PyTorch tensors. Inputs exceeding
    max_input_tokens are rejected rather than silently truncated.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    text = text.strip()
    if not text:
        raise ValueError("Please provide non-empty text.")

    prompt = "grammar: " + text
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)

    token_count = int(encoded["input_ids"].shape[-1])
    if token_count > max_input_tokens:
        raise ValueError(
            f"This sentence contains {token_count} tokens, exceeding the "
            f"configured limit of {max_input_tokens}. Split or shorten the "
            "sentence before correction; the input was not truncated."
        )

    device = _model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

    corrected = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    ).strip()

    if capitalize_initial:
        corrected = _capitalize_sentence_start(corrected)

    return corrected


def correct_paragraph(
    text: str,
    tokenizer,
    model: torch.nn.Module,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    num_beams: int = DEFAULT_NUM_BEAMS,
    preserve_paragraph_breaks: bool = True,
    capitalize_initial: bool = True,
) -> str:
    """
    Segment text with PySBD, correct each sentence, then reconstruct paragraphs.

    Blank-line-separated paragraphs remain separate. Single line breaks inside
    a paragraph are treated as line wrapping. Sentence segmentation remains
    heuristic and should be tested on the intended document types.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    text = text.strip()
    if not text:
        raise ValueError("Please provide a non-empty paragraph.")

    segmenter = pysbd.Segmenter(language="en", clean=False)
    paragraphs = re.split(r"\n[ \t]*\n+", text)
    corrected_paragraphs: List[str] = []

    for paragraph_index, paragraph in enumerate(paragraphs, start=1):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue

        sentences = segmenter.segment(paragraph)
        corrected_sentences: List[str] = []

        for sentence_index, sentence in enumerate(sentences, start=1):
            sentence = sentence.strip()
            if not sentence:
                continue

            try:
                corrected = correct_grammar(
                    sentence,
                    tokenizer=tokenizer,
                    model=model,
                    max_input_tokens=max_input_tokens,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    capitalize_initial=capitalize_initial,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Paragraph {paragraph_index}, sentence {sentence_index}: {exc}"
                ) from exc

            corrected_sentences.append(corrected)

        if corrected_sentences:
            corrected_paragraphs.append(" ".join(corrected_sentences))

    separator = "\n\n" if preserve_paragraph_breaks else " "
    return separator.join(corrected_paragraphs)


_ALIGN_TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", flags=re.UNICODE)


def _is_punctuation_token(token: str) -> bool:
    """Return True for a punctuation/symbol token rather than a word token."""
    return bool(token) and not any(ch.isalnum() or ch == "_" for ch in token)


def _join_token_text(text: str, matches, start: int, end: int) -> str:
    """Get the exact corrected substring covered by token indexes [start, end)."""
    if start >= end:
        return ""
    return text[matches[start].start():matches[end - 1].end()]


def get_text_level_changes(original: str, corrected: str) -> List[Dict[str, Any]]:
    """
    Align original and corrected text at both word and punctuation level.

    Capitalization is case-sensitive. Punctuation insertions are represented
    as zero-width changes so the UI can show the suggested mark at its inferred
    location without replacing the user's original text.

    Changes are model suggestions, not independently verified grammar labels.
    """
    if not isinstance(original, str) or not isinstance(corrected, str):
        raise TypeError("Both original and corrected values must be strings.")

    original_matches = list(_ALIGN_TOKEN_PATTERN.finditer(original))
    corrected_matches = list(_ALIGN_TOKEN_PATTERN.finditer(corrected))
    original_tokens = [m.group(0) for m in original_matches]
    corrected_tokens = [m.group(0) for m in corrected_matches]

    matcher = SequenceMatcher(
        a=original_tokens,
        b=corrected_tokens,
        autojunk=False,
    )

    changes: List[Dict[str, Any]] = []

    def add_word_span(old_word_matches, suggestion: str, change_type: str):
        if not old_word_matches:
            return
        start = old_word_matches[0].start()
        end = old_word_matches[-1].end()
        changes.append({
            "kind": "word",
            "start": start,
            "end": end,
            "original": original[start:end],
            "suggestion": suggestion,
            "change_type": change_type,
        })

    def add_punctuation_span(match, suggestion: str, change_type: str):
        changes.append({
            "kind": "punctuation",
            "start": match.start(),
            "end": match.end(),
            "original": match.group(0),
            "suggestion": suggestion,
            "change_type": change_type,
        })

    def add_word_insertion(anchor, suggestion: str, position: str):
        if anchor is None or not suggestion:
            return
        changes.append({
            "kind": "word_insertion",
            "start": anchor.start(),
            "end": anchor.end(),
            "original": anchor.group(0),
            "suggestion": suggestion,
            "change_type": "insert",
            "position": position,
        })

    def add_punctuation_insertion(position: int, mark: str):
        changes.append({
            "kind": "punctuation_insertion",
            "start": position,
            "end": position,
            "original": "",
            "suggestion": mark,
            "change_type": "insert",
        })

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        old_block = list(enumerate(original_matches[i1:i2], start=i1))
        new_block = list(enumerate(corrected_matches[j1:j2], start=j1))
        old_words = [(idx, m) for idx, m in old_block if not _is_punctuation_token(m.group(0))]
        new_words = [(idx, m) for idx, m in new_block if not _is_punctuation_token(m.group(0))]
        old_punctuation = [(idx, m) for idx, m in old_block if _is_punctuation_token(m.group(0))]
        new_punctuation = [(idx, m) for idx, m in new_block if _is_punctuation_token(m.group(0))]

        # Build a local word alignment for replacement blocks. This also maps
        # corrected words back to original words for positioning punctuation.
        old_word_values = [m.group(0) for _, m in old_words]
        new_word_values = [m.group(0) for _, m in new_words]
        local_word_matcher = SequenceMatcher(
            a=old_word_values,
            b=new_word_values,
            autojunk=False,
        )
        corrected_local_to_original: Dict[int, int] = {}

        for word_tag, wi1, wi2, wj1, wj2 in local_word_matcher.get_opcodes():
            if word_tag == "equal":
                for offset in range(wi2 - wi1):
                    corrected_local_to_original[wj1 + offset] = wi1 + offset
            elif word_tag == "replace":
                for old_local, new_local in zip(range(wi1, wi2), range(wj1, wj2)):
                    corrected_local_to_original[new_local] = old_local

            suggestion = " ".join(new_word_values[wj1:wj2])

            if word_tag in ("replace", "delete") and wi1 < wi2:
                add_word_span(
                    [m for _, m in old_words[wi1:wi2]],
                    suggestion,
                    word_tag,
                )
            elif word_tag == "insert" and wj1 < wj2 and old_words:
                if wi1 < len(old_words):
                    anchor = old_words[wi1][1]
                    position = "before"
                else:
                    anchor = old_words[-1][1]
                    position = "after"
                add_word_insertion(anchor, suggestion, position)

        # Align punctuation only within this changed token block. Pair changed
        # punctuation individually so an edit like ',' -> '.' never underlines
        # the words between two punctuation marks.
        old_punc_values = [m.group(0) for _, m in old_punctuation]
        new_punc_values = [m.group(0) for _, m in new_punctuation]
        local_punc_matcher = SequenceMatcher(
            a=old_punc_values,
            b=new_punc_values,
            autojunk=False,
        )

        def punctuation_insertion_offset(new_punc_match) -> int:
            # Find the closest preceding corrected word that maps to an
            # original word; otherwise use the next mapped word.
            previous = []
            for new_local, old_local in corrected_local_to_original.items():
                candidate = new_words[new_local][1]
                if candidate.end() <= new_punc_match.start():
                    previous.append((candidate.end(), old_local))
            if previous:
                _, old_local = max(previous, key=lambda item: item[0])
                return old_words[old_local][1].end()

            following = []
            for new_local, old_local in corrected_local_to_original.items():
                candidate = new_words[new_local][1]
                if candidate.start() >= new_punc_match.end():
                    following.append((candidate.start(), old_local))
            if following:
                _, old_local = min(following, key=lambda item: item[0])
                return old_words[old_local][1].start()

            # No aligned word inside this block: anchor at the surrounding
            # original token boundary.
            if i1 > 0:
                return original_matches[i1 - 1].end()
            if old_words:
                return old_words[-1][1].end()
            if i2 < len(original_matches):
                return original_matches[i2].start()
            return len(original)

        for punc_tag, pi1, pi2, pj1, pj2 in local_punc_matcher.get_opcodes():
            if punc_tag == "equal":
                continue

            if punc_tag == "replace":
                paired = min(pi2 - pi1, pj2 - pj1)
                for offset in range(paired):
                    old_match = old_punctuation[pi1 + offset][1]
                    new_mark = new_punc_values[pj1 + offset]
                    add_punctuation_span(old_match, new_mark, "replace")
                for old_local in range(pi1 + paired, pi2):
                    add_punctuation_span(
                        old_punctuation[old_local][1], "", "delete"
                    )
                for new_local in range(pj1 + paired, pj2):
                    new_match = new_punctuation[new_local][1]
                    add_punctuation_insertion(
                        punctuation_insertion_offset(new_match),
                        new_match.group(0),
                    )

            elif punc_tag == "delete":
                for old_local in range(pi1, pi2):
                    add_punctuation_span(
                        old_punctuation[old_local][1], "", "delete"
                    )

            elif punc_tag == "insert":
                for new_local in range(pj1, pj2):
                    new_match = new_punctuation[new_local][1]
                    add_punctuation_insertion(
                        punctuation_insertion_offset(new_match),
                        new_match.group(0),
                    )

        # If a top-level inserted block contains punctuation but no old words,
        # the local alignment above still creates punctuation suggestions. If
        # it contains only inserted words and no old words, anchor globally.
        if tag == "insert" and new_words and not old_words:
            next_word = next(
                (m for idx, m in enumerate(original_matches[i1:], start=i1)
                 if not _is_punctuation_token(m.group(0))),
                None,
            )
            previous_word = next(
                (m for m in reversed(original_matches[:i1])
                 if not _is_punctuation_token(m.group(0))),
                None,
            )
            anchor = next_word or previous_word
            position = "before" if next_word is not None else "after"
            suggestion = " ".join(new_word_values)
            add_word_insertion(anchor, suggestion, position)

    return changes


def get_word_level_changes(original: str, corrected: str) -> List[Dict[str, Any]]:
    """Backward-compatible name; returns word and punctuation changes."""
    return get_text_level_changes(original, corrected)


def render_word_highlights_html(original: str, corrected: str) -> str:
    """
    Render the ORIGINAL text with red wavy underlines for model-suggested edits.

    - Replaced/deleted words and existing punctuation are underlined in place.
    - Missing words underline a nearby original word with an insertion hint.
    - Missing punctuation is shown as a red-underlined suggested mark at the
      inferred insertion point, e.g. a proposed comma after "traffic".

    User-supplied text is HTML-escaped before rendering.
    """
    if not isinstance(original, str) or not isinstance(corrected, str):
        raise TypeError("Both original and corrected values must be strings.")

    changes = get_text_level_changes(original, corrected)
    span_hints: Dict[tuple, List[str]] = {}
    span_kinds: Dict[tuple, set] = {}
    punctuation_insertions: Dict[int, List[str]] = {}

    for change in changes:
        kind = change["kind"]
        if kind == "punctuation_insertion":
            punctuation_insertions.setdefault(change["start"], []).append(
                change["suggestion"]
            )
            continue

        key = (change["start"], change["end"])
        span_kinds.setdefault(key, set()).add(kind)

        if kind == "word_insertion":
            hint = f"Suggested word {change['position']}: {change['suggestion']}"
        elif kind == "punctuation":
            if change["change_type"] == "delete":
                hint = f"Suggested punctuation removal: {change['original']}"
            else:
                hint = f"Suggested punctuation: {change['suggestion'] or '(remove)'}"
        elif change["change_type"] == "delete":
            hint = "Possible word removal"
        else:
            hint = f"Suggested replacement: {change['suggestion'] or '(remove)'}"

        span_hints.setdefault(key, []).append(hint)

    spans = sorted(
        ((start, end, hints) for (start, end), hints in span_hints.items()),
        key=lambda item: (item[0], item[1]),
    )

    # Avoid rendering overlapping source spans more than once.
    non_overlapping = []
    last_end = -1
    for start, end, hints in spans:
        if start < last_end:
            continue
        non_overlapping.append((start, end, hints))
        last_end = end

    pieces: List[str] = []
    cursor = 0

    def emit_punctuation_at(position: int) -> None:
        for mark in punctuation_insertions.get(position, []):
            visible = html.escape(mark)
            title = html.escape(f"Suggested punctuation: {mark}", quote=True)
            pieces.append(
                f'<span class="grammarai-punctuation-insertion" '
                f'title="{title}">{visible}</span>'
            )

    for start, end, hints in non_overlapping:
        for position in sorted(
            p for p in punctuation_insertions if cursor <= p <= start
        ):
            pieces.append(html.escape(original[cursor:position]))
            cursor = position
            emit_punctuation_at(position)

        pieces.append(html.escape(original[cursor:start]))
        title = html.escape(" | ".join(dict.fromkeys(hints)), quote=True)
        content = html.escape(original[start:end])
        kinds = span_kinds.get((start, end), set())
        css_class = (
            "grammarai-changed-punctuation"
            if "punctuation" in kinds and "word" not in kinds
            else "grammarai-changed-word"
        )
        pieces.append(
            f'<span class="{css_class}" title="{title}">{content}</span>'
        )
        cursor = end

    for position in sorted(p for p in punctuation_insertions if p >= cursor):
        pieces.append(html.escape(original[cursor:position]))
        cursor = position
        emit_punctuation_at(position)

    pieces.append(html.escape(original[cursor:]))
    highlighted = "".join(pieces)

    return (
        "<style>"
        ".grammarai-highlight-container{white-space:pre-wrap;line-height:1.9;"
        "font-family:inherit;}"
        ".grammarai-changed-word,.grammarai-changed-punctuation,"
        ".grammarai-punctuation-insertion{"
        "text-decoration-line:underline;"
        "text-decoration-style:wavy;"
        "text-decoration-color:#dc2626;"
        "text-decoration-thickness:2px;"
        "text-underline-offset:3px;"
        "cursor:help;"
        "}"
        ".grammarai-punctuation-insertion{color:#dc2626;font-weight:600;}"
        "</style>"
        f'<div class="grammarai-highlight-container">{highlighted}</div>'
    )


def validate_output(
    original: str,
    corrected: str,
    suspicious_shortening_ratio: float = 0.5,
    min_original_words_for_ratio_check: int = 12,
) -> Dict[str, Any]:
    """
    Run basic structural checks only; this cannot verify semantic equivalence.
    A large word-count reduction is a warning for human review, not proof.
    """
    if not isinstance(original, str):
        raise TypeError("Original text must be a string.")
    if not isinstance(corrected, str):
        raise TypeError("Corrected text must be a string.")

    original_clean = original.strip()
    corrected_clean = corrected.strip()
    original_words = len(original_clean.split())
    corrected_words = len(corrected_clean.split())

    errors: List[str] = []
    warnings: List[str] = []

    if not original_clean:
        errors.append("Original text is empty.")
    if not corrected_clean:
        errors.append("The model returned empty output.")

    if (
        original_words >= min_original_words_for_ratio_check
        and corrected_words < original_words * suspicious_shortening_ratio
    ):
        warnings.append(
            "Output is substantially shorter than input. Review for omitted information."
        )

    if original_clean and corrected_clean == original_clean:
        warnings.append(
            "Output is unchanged. This may be correct for clean text or indicate an uncorrected error."
        )

    return {
        "is_usable": not errors,
        "errors": errors,
        "warnings": warnings,
        "original_word_count": original_words,
        "corrected_word_count": corrected_words,
    }
