"""Core, dependency-free primitives for CAGRO.

This module implements the parts of *Confidence-Aware Group-Relative
Optimization* that do not depend on PyTorch or Transformers:

* strict semantic-span parsing for ``context``, ``think``, and ``answer``;
* fail-closed character-to-token span mapping; and
* the serial task-validity and group-relative support gates from Eqs. (3)-(5).

Keeping these rules independent from the trainer makes the paper-critical
logic unit-testable without loading a multimodal model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Optional, Sequence


SEMANTIC_TAGS = ("context", "think", "answer")


@dataclass(frozen=True)
class CharacterSpan:
    """Half-open character interval for the body of one semantic tag."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid character span [{self.start}, {self.end})")


@dataclass(frozen=True)
class CAGROGateConfig:
    """Paper defaults for CAGRO's serial dual gate."""

    answer_threshold: float = 0.5
    context_threshold: float = 0.3
    reasoning_threshold: float = 0.4
    support_upper_bound: float = 0.8
    support_tolerance: float = 0.05
    bonus_coefficient: float = 0.2

    def __post_init__(self) -> None:
        for name in ("answer_threshold", "context_threshold", "reasoning_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not 0.0 < self.support_upper_bound <= 1.0:
            raise ValueError(
                "support_upper_bound must be in (0, 1], "
                f"got {self.support_upper_bound}"
            )
        if self.support_tolerance < 0.0:
            raise ValueError(
                f"support_tolerance must be non-negative, got {self.support_tolerance}"
            )
        if self.bonus_coefficient < 0.0:
            raise ValueError(
                f"bonus_coefficient must be non-negative, got {self.bonus_coefficient}"
            )


@dataclass(frozen=True)
class CAGROGateResult:
    """Per-candidate diagnostics and bonuses produced by Eqs. (3)-(5)."""

    bonuses: tuple[float, ...]
    task_valid: tuple[bool, ...]
    support_valid: tuple[bool, ...]
    relative_pass: tuple[bool, ...]
    clipped_support: tuple[Optional[float], ...]
    group_reference: Optional[float]


def find_unique_nonempty_tag_span(text: str, tag: str) -> Optional[CharacterSpan]:
    """Return a unique, non-empty tag body or ``None``.

    The match is deliberately case-sensitive: the response protocol prescribes
    exact lower-case structural tags. No fallback span is fabricated.
    """

    if not tag or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", tag):
        raise ValueError(f"Invalid tag name: {tag!r}")

    text = str(text)
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    if text.count(open_tag) != 1 or text.count(close_tag) != 1:
        return None

    open_start = text.find(open_tag)
    body_start = open_start + len(open_tag)
    body_end = text.find(close_tag, body_start)
    if body_end < body_start or not text[body_start:body_end].strip():
        return None
    return CharacterSpan(body_start, body_end)


def find_ordered_semantic_spans(text: str) -> Optional[dict[str, CharacterSpan]]:
    """Parse one unique, non-empty, ordered body for every CAGRO tag.

    Leading/trailing free text is tolerated here because this parser is used to
    preserve local *base* supervision for otherwise malformed candidates. The
    format reward separately enforces the exact response grammar. Returned
    spans are guaranteed to be mutually disjoint.
    """

    text = str(text)
    spans: dict[str, CharacterSpan] = {}
    previous_close_end = -1

    for tag in SEMANTIC_TAGS:
        span = find_unique_nonempty_tag_span(text, tag)
        if span is None:
            return None
        open_start = text.find(f"<{tag}>")
        close_end = text.find(f"</{tag}>", span.end) + len(f"</{tag}>")
        if open_start < previous_close_end or close_end <= span.end:
            return None
        spans[tag] = span
        previous_close_end = close_end

    return spans


def parse_strict_cagro_response(text: str) -> Optional[dict[str, CharacterSpan]]:
    """Parse the exact three-span response grammar from Eq. (1)."""

    text = str(text)
    spans = find_ordered_semantic_spans(text)
    if spans is None:
        return None

    strict_pattern = re.compile(
        r"\s*<context>(?P<context>.*?)</context>\s*"
        r"<think>(?P<think>.*?)</think>\s*"
        r"<answer>(?P<answer>.*?)</answer>\s*",
        re.DOTALL,
    )
    match = strict_pattern.fullmatch(text)
    if match is None or any(not match.group(tag).strip() for tag in SEMANTIC_TAGS):
        return None

    return {
        tag: CharacterSpan(*match.span(tag))
        for tag in SEMANTIC_TAGS
    }


def extract_unique_tag_text(text: str, tag: str) -> str:
    """Extract a unique non-empty tag body, returning ``""`` on failure."""

    span = find_unique_nonempty_tag_span(text, tag)
    if span is None:
        return ""
    return str(text)[span.start:span.end].strip()


def map_character_span_to_token_mask(
    decoded_prefixes: Sequence[str],
    span: CharacterSpan,
) -> Optional[tuple[bool, ...]]:
    """Map a character body to tokens wholly contained inside it.

    ``decoded_prefixes`` must contain the decoded empty prefix followed by the
    decoded prefix after every generated token. A token that straddles a tag
    boundary is excluded, because the paper defines semantic indices strictly
    inside tag pairs. Non-monotonic decoding or an empty mapping fails closed.
    """

    if len(decoded_prefixes) < 2:
        return None
    lengths = [len(prefix) for prefix in decoded_prefixes]
    if any(right < left for left, right in zip(lengths, lengths[1:])):
        return None
    if span.end > lengths[-1]:
        return None

    mask = tuple(
        token_start >= span.start
        and token_end <= span.end
        and token_end > token_start
        for token_start, token_end in zip(lengths, lengths[1:])
    )
    return mask if any(mask) else None


def _validate_equal_lengths(named_values: Mapping[str, Sequence[object]]) -> int:
    lengths = {name: len(values) for name, values in named_values.items()}
    if not lengths:
        raise ValueError("At least one candidate sequence is required")
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"Candidate sequences must have equal lengths: {lengths}")
    size = unique_lengths.pop()
    if size == 0:
        raise ValueError("A CAGRO group must contain at least one candidate")
    return size


def compute_cagro_gate(
    *,
    format_rewards: Sequence[float],
    answer_rewards: Sequence[float],
    context_rewards: Sequence[float],
    reasoning_rewards: Sequence[float],
    answer_span_valid: Sequence[bool],
    answer_support: Sequence[Optional[float]],
    config: CAGROGateConfig = CAGROGateConfig(),
) -> CAGROGateResult:
    """Apply CAGRO's task-validity gate and relative-support gate.

    This is a direct implementation of Eqs. (3)-(5). Gate I controls only
    eligibility for the bounded bonus; it never removes a candidate from base
    GRPO supervision.
    """

    size = _validate_equal_lengths(
        {
            "format_rewards": format_rewards,
            "answer_rewards": answer_rewards,
            "context_rewards": context_rewards,
            "reasoning_rewards": reasoning_rewards,
            "answer_span_valid": answer_span_valid,
            "answer_support": answer_support,
        }
    )

    task_valid: list[bool] = []
    support_valid: list[bool] = []
    clipped_support: list[Optional[float]] = []

    for index in range(size):
        rewards = (
            float(format_rewards[index]),
            float(answer_rewards[index]),
            float(context_rewards[index]),
            float(reasoning_rewards[index]),
        )
        rewards_finite = all(math.isfinite(value) for value in rewards)
        valid = (
            bool(answer_span_valid[index])
            and rewards_finite
            and rewards[0] == 1.0
            and rewards[1] >= config.answer_threshold
            and rewards[2] >= config.context_threshold
            and rewards[3] >= config.reasoning_threshold
        )
        task_valid.append(valid)

        raw_support = answer_support[index]
        finite_support = (
            raw_support is not None
            and math.isfinite(float(raw_support))
            and float(raw_support) >= 0.0
        )
        support_valid.append(finite_support)
        clipped_support.append(
            min(float(raw_support), config.support_upper_bound)
            if finite_support
            else None
        )

    eligible_indices = [
        index
        for index in range(size)
        if task_valid[index] and support_valid[index]
    ]
    relative_pass = [False] * size
    bonuses = [0.0] * size

    # Eq. (5) fails closed for a singleton valid set.
    if len(eligible_indices) < 2:
        return CAGROGateResult(
            bonuses=tuple(bonuses),
            task_valid=tuple(task_valid),
            support_valid=tuple(support_valid),
            relative_pass=tuple(relative_pass),
            clipped_support=tuple(clipped_support),
            group_reference=None,
        )

    group_reference = sum(
        clipped_support[index] for index in eligible_indices  # type: ignore[arg-type]
    ) / len(eligible_indices)
    threshold = group_reference - config.support_tolerance

    for index in eligible_indices:
        if clipped_support[index] >= threshold:  # type: ignore[operator]
            relative_pass[index] = True
            bonuses[index] = config.bonus_coefficient * float(answer_rewards[index])

    return CAGROGateResult(
        bonuses=tuple(bonuses),
        task_valid=tuple(task_valid),
        support_valid=tuple(support_valid),
        relative_pass=tuple(relative_pass),
        clipped_support=tuple(clipped_support),
        group_reference=group_reference,
    )
