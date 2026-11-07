"""Manage accept/reject decisions for GrammarAI model suggestions.

Consumes change dictionaries returned by model_utils.get_text_level_changes().
All offsets refer to the original input string. No model inference is done here.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


class CorrectionManagerError(ValueError):
    """Base error for invalid correction suggestions."""


class CorrectionConflictError(CorrectionManagerError):
    """Raised when accepted edits overlap and cannot safely be combined."""


class CorrectionManager:
    """Track multiple suggestions and reconstruct text from accepted edits."""

    VALID_STATUSES = {"pending", "accepted", "rejected"}

    def __init__(self, original_text: str, changes: Sequence[Dict[str, Any]]):
        if not isinstance(original_text, str):
            raise TypeError("original_text must be a string.")
        if not original_text.strip():
            raise ValueError("original_text cannot be empty.")
        if changes is None:
            raise TypeError("changes must be a sequence of dictionaries.")

        self.original_text = original_text
        self._suggestions: List[Dict[str, Any]] = []

        for suggestion_id, raw in enumerate(changes, start=1):
            if not isinstance(raw, dict):
                raise TypeError(f"Suggestion {suggestion_id} must be a dictionary.")
            item = deepcopy(raw)
            self._validate_change(item, suggestion_id)
            item["id"] = suggestion_id
            item["status"] = "pending"
            self._suggestions.append(item)

    def _validate_change(self, item: Dict[str, Any], suggestion_id: int) -> None:
        required = {"kind", "start", "end", "original", "suggestion", "change_type"}
        missing = required - item.keys()
        if missing:
            raise CorrectionManagerError(
                f"Suggestion {suggestion_id} is missing: {sorted(missing)}"
            )

        start, end = item["start"], item["end"]
        if not isinstance(start, int) or not isinstance(end, int):
            raise CorrectionManagerError("Suggestion offsets must be integers.")
        if start < 0 or end < start or end > len(self.original_text):
            raise CorrectionManagerError(f"Suggestion {suggestion_id} has invalid offsets.")

        kind = item["kind"]
        change_type = item["change_type"]
        if kind not in {"word", "punctuation", "word_insertion", "punctuation_insertion"}:
            raise CorrectionManagerError(f"Unsupported suggestion kind: {kind!r}")
        if change_type not in {"replace", "delete", "insert"}:
            raise CorrectionManagerError(f"Unsupported change type: {change_type!r}")

        if kind == "punctuation_insertion":
            if change_type != "insert" or start != end:
                raise CorrectionManagerError("Punctuation insertion needs a zero-width span.")
        elif kind == "word_insertion":
            if change_type != "insert" or start == end:
                raise CorrectionManagerError("Word insertion needs a non-empty anchor span.")
            if item.get("position") not in {"before", "after"}:
                raise CorrectionManagerError("Word insertion position must be 'before' or 'after'.")
        else:
            if change_type == "insert" or start == end:
                raise CorrectionManagerError(
                    "Use a dedicated insertion kind for inserted words or punctuation."
                )

        expected = item["original"]
        if not isinstance(expected, str) or not isinstance(item["suggestion"], str):
            raise CorrectionManagerError("original and suggestion must be strings.")

        # A word insertion uses the original word as an anchor, not as a replaced span.
        if start != end and self.original_text[start:end] != expected:
            raise CorrectionManagerError(
                f"Suggestion {suggestion_id} is stale: source span "
                f"{self.original_text[start:end]!r} does not match {expected!r}."
            )

    @property
    def suggestions(self) -> List[Dict[str, Any]]:
        """Defensive copy of suggestions, including ID and review status."""
        return deepcopy(self._suggestions)

    def get_suggestions(self) -> List[Dict[str, Any]]:
        return self.suggestions

    def _get(self, suggestion_id: int) -> Dict[str, Any]:
        for item in self._suggestions:
            if item["id"] == suggestion_id:
                return item
        raise KeyError(f"Unknown suggestion ID: {suggestion_id}")

    def set_status(self, suggestion_id: int, status: str) -> None:
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Status must be one of {sorted(self.VALID_STATUSES)}.")
        self._get(suggestion_id)["status"] = status

    def accept(self, suggestion_id: int) -> None:
        self.set_status(suggestion_id, "accepted")

    def reject(self, suggestion_id: int) -> None:
        self.set_status(suggestion_id, "rejected")

    def set_pending(self, suggestion_id: int) -> None:
        self.set_status(suggestion_id, "pending")

    def accept_all(self) -> None:
        for item in self._suggestions:
            item["status"] = "accepted"

    def reject_all(self) -> None:
        for item in self._suggestions:
            item["status"] = "rejected"

    def reset_all(self) -> None:
        for item in self._suggestions:
            item["status"] = "pending"

    def summary(self) -> Dict[str, int]:
        counts = {"total": len(self._suggestions), "pending": 0,
                  "accepted": 0, "rejected": 0}
        for item in self._suggestions:
            counts[item["status"]] += 1
        return counts

    def _accepted_operations(self):
        # Replacement/deletion spans: (start, end, replacement, suggestion_id)
        spans: List[Tuple[int, int, str, int]] = []
        # Insertions: source offset -> [(insertion text, suggestion ID)]
        insertions: Dict[int, List[Tuple[str, int]]] = {}

        for item in self._suggestions:
            if item["status"] != "accepted":
                continue

            sid = item["id"]
            kind = item["kind"]
            start, end = item["start"], item["end"]
            suggestion = item["suggestion"]

            if kind == "punctuation_insertion":
                insertions.setdefault(start, []).append((suggestion, sid))
            elif kind == "word_insertion":
                if item["position"] == "before":
                    offset, text = start, suggestion.strip() + " "
                else:
                    offset, text = end, " " + suggestion.strip()
                insertions.setdefault(offset, []).append((text, sid))
            else:
                replacement = suggestion if item["change_type"] == "replace" else ""
                spans.append((start, end, replacement, sid))

        spans.sort(key=lambda op: (op[0], op[1], op[3]))

        previous = None
        for current in spans:
            if previous is not None and current[0] < previous[1]:
                raise CorrectionConflictError(
                    f"Accepted suggestions #{previous[3]} and #{current[3]} "
                    "overlap. They must be reviewed as a single correction group."
                )
            previous = current

        for offset, insert_items in insertions.items():
            for start, end, _replacement, span_id in spans:
                if start < offset < end:
                    ids = [sid for _, sid in insert_items]
                    raise CorrectionConflictError(
                        f"Insertion suggestion(s) {ids} overlap replacement "
                        f"suggestion #{span_id}."
                    )
            insert_items.sort(key=lambda pair: pair[1])

        return spans, insertions

    def apply_accepted(self) -> str:
        """Apply accepted suggestions; pending/rejected edits remain original."""
        spans, insertions = self._accepted_operations()
        spans_by_start: Dict[int, Tuple[int, str, int]] = {}

        for start, end, replacement, sid in spans:
            if start in spans_by_start:
                other = spans_by_start[start][2]
                raise CorrectionConflictError(
                    f"Suggestions #{other} and #{sid} share the same start offset."
                )
            spans_by_start[start] = (end, replacement, sid)

        positions = sorted(set(spans_by_start) | set(insertions))
        output: List[str] = []
        cursor = 0

        for position in positions:
            if position < cursor:
                raise CorrectionConflictError(
                    f"An accepted edit at offset {position} overlaps an earlier edit."
                )

            output.append(self.original_text[cursor:position])

            for insert_text, _sid in insertions.get(position, []):
                output.append(insert_text)

            if position in spans_by_start:
                end, replacement, _sid = spans_by_start[position]
                output.append(replacement)
                cursor = end
            else:
                cursor = position

        output.append(self.original_text[cursor:])
        return "".join(output)

    def apply(self) -> str:
        """Convenience alias for apply_accepted()."""
        return self.apply_accepted()

    def apply_all(self) -> str:
        self.accept_all()
        return self.apply_accepted()

    def reject_all_and_restore(self) -> str:
        self.reject_all()
        return self.original_text
