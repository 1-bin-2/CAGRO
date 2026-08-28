from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from open_r1.cagro import (  # noqa: E402
    CAGROGateConfig,
    CharacterSpan,
    compute_cagro_gate,
    find_ordered_semantic_spans,
    find_unique_nonempty_tag_span,
    map_character_span_to_token_mask,
    parse_strict_cagro_response,
)


VALID_RESPONSE = (
    "<context>visible evidence</context>"
    "<think>reason from evidence</think>"
    "<answer>B</answer>"
)


class SemanticSpanTests(unittest.TestCase):
    def test_strict_parser_accepts_exact_three_span_protocol(self):
        spans = parse_strict_cagro_response(VALID_RESPONSE)
        self.assertIsNotNone(spans)
        assert spans is not None
        self.assertEqual(VALID_RESPONSE[spans["context"].start : spans["context"].end], "visible evidence")
        self.assertEqual(VALID_RESPONSE[spans["think"].start : spans["think"].end], "reason from evidence")
        self.assertEqual(VALID_RESPONSE[spans["answer"].start : spans["answer"].end], "B")

    def test_strict_parser_fails_closed_on_duplicate_empty_or_wrong_order(self):
        malformed = (
            VALID_RESPONSE + "<answer>C</answer>",
            "<context> </context><think>x</think><answer>A</answer>",
            "<think>x</think><context>y</context><answer>A</answer>",
            "prefix " + VALID_RESPONSE,
        )
        for response in malformed:
            with self.subTest(response=response):
                self.assertIsNone(parse_strict_cagro_response(response))

    def test_local_span_parser_preserves_valid_bodies_when_format_is_invalid(self):
        response = "prefix " + VALID_RESPONSE + " suffix"
        spans = find_ordered_semantic_spans(response)
        self.assertIsNotNone(spans)

    def test_unique_answer_span_rejects_missing_or_duplicate_close(self):
        self.assertIsNone(find_unique_nonempty_tag_span("<answer>A", "answer"))
        self.assertIsNone(
            find_unique_nonempty_tag_span("<answer>A</answer></answer>", "answer")
        )

    def test_character_to_token_mask_excludes_boundary_tokens(self):
        # Token intervals are [0,2), [2,4), [4,6); only the middle token lies
        # wholly inside [2,5).
        mask = map_character_span_to_token_mask(
            ["", "ab", "abcd", "abcdef"], CharacterSpan(2, 5)
        )
        self.assertEqual(mask, (False, True, False))

    def test_character_to_token_mask_fails_on_non_monotonic_decode(self):
        self.assertIsNone(
            map_character_span_to_token_mask(["", "ab", "a"], CharacterSpan(0, 1))
        )


class GateTests(unittest.TestCase):
    def _gate(self, *, supports, answer=(1.0, 0.8, 0.7, 0.6), **overrides):
        values = {
            "format_rewards": (1.0, 1.0, 1.0, 1.0),
            "answer_rewards": answer,
            "context_rewards": (0.8, 0.8, 0.8, 0.8),
            "reasoning_rewards": (0.8, 0.8, 0.8, 0.8),
            "answer_span_valid": (True, True, True, True),
            "answer_support": supports,
        }
        values.update(overrides)
        return compute_cagro_gate(**values)

    def test_dual_gate_uses_valid_set_mean_and_mean_minus_delta(self):
        result = self._gate(supports=(0.8, 0.7, 0.1, 0.8))
        self.assertAlmostEqual(result.group_reference, 0.6)
        self.assertEqual(result.relative_pass, (True, True, False, True))
        for actual, expected in zip(result.bonuses, (0.2, 0.16, 0.0, 0.12)):
            self.assertAlmostEqual(actual, expected)

    def test_task_invalid_high_support_is_excluded_from_reference(self):
        result = self._gate(
            supports=(0.2, 0.4, 1.0, 0.3),
            context_rewards=(0.8, 0.8, 0.0, 0.8),
        )
        # The invalid third sample's clipped 0.8 support must not lift the mean.
        self.assertAlmostEqual(result.group_reference, 0.3)
        self.assertEqual(result.relative_pass, (False, True, False, True))
        self.assertEqual(result.bonuses[2], 0.0)

    def test_singleton_valid_set_fails_closed(self):
        result = self._gate(
            supports=(0.8, math.nan, None, -1.0),
            format_rewards=(1.0, 0.0, 0.0, 0.0),
        )
        self.assertIsNone(result.group_reference)
        self.assertEqual(result.bonuses, (0.0, 0.0, 0.0, 0.0))

    def test_bonus_is_answer_scaled_and_support_is_only_binary_eligibility(self):
        result = self._gate(supports=(0.8, 0.8, 0.8, 0.8))
        for actual, expected in zip(result.bonuses, (0.2, 0.16, 0.14, 0.12)):
            self.assertAlmostEqual(actual, expected)

    def test_thresholds_are_inclusive(self):
        result = self._gate(
            supports=(0.5, 0.5, None, None),
            answer=(0.5, 0.5, 0.0, 0.0),
            context_rewards=(0.3, 0.3, 0.0, 0.0),
            reasoning_rewards=(0.4, 0.4, 0.0, 0.0),
        )
        self.assertEqual(result.task_valid, (True, True, False, False))
        self.assertEqual(result.relative_pass, (True, True, False, False))

    def test_nonfinite_reward_fails_gate_one(self):
        result = self._gate(
            supports=(0.8, 0.8, 0.8, 0.8),
            reasoning_rewards=(0.8, math.inf, 0.8, 0.8),
        )
        self.assertFalse(result.task_valid[1])
        self.assertEqual(result.bonuses[1], 0.0)

    def test_group_inputs_must_have_equal_lengths(self):
        with self.assertRaises(ValueError):
            compute_cagro_gate(
                format_rewards=(1.0,),
                answer_rewards=(1.0, 1.0),
                context_rewards=(1.0,),
                reasoning_rewards=(1.0,),
                answer_span_valid=(True,),
                answer_support=(0.8,),
                config=CAGROGateConfig(),
            )


if __name__ == "__main__":
    unittest.main()
