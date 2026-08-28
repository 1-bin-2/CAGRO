# Paper alignment audit

This document distinguishes requirements stated by the CAGRO paper from
implementation choices needed where the paper is silent.

## Directly implemented requirements

| Paper requirement | Implementation |
| --- | --- |
| Exact ordered `<context>`, `<think>`, `<answer>` response protocol | `open_r1.cagro.parse_strict_cagro_response` and `QwenOmniModule.format_reward` |
| Semantic token sets contain tag bodies only and are mutually disjoint | `VLMGRPOTrainer._get_semantic_span_masks` |
| Four `[0,1]` base signals; no fifth consistency reward | `GRPOScriptArguments`, reward registry, and CAGRO constructor validation |
| Frozen RL-start reference with no gradient or update | Reference construction/freezing plus rejection of EMA and sync modes in CAGRO |
| Answer support is the answer-token geometric mean under the full prefix | Teacher-forced frozen-reference log probabilities and answer-body mask |
| Unique, non-empty answer tags, mappable character interval, finite log probabilities | Fail-closed span mapping before support eligibility |
| Upper clipping only, `tau=0.8` | `CAGROGateConfig.support_upper_bound` |
| Gate I thresholds `(answer=0.5, context=0.3, reasoning=0.4)` | `compute_cagro_gate` |
| Gate II mean over task-valid finite-support candidates only | `compute_cagro_gate` |
| Pass condition `support >= mean - 0.05` | `compute_cagro_gate` |
| Fewer than two eligible candidates receive no bonus | Explicit singleton fail-closed branch |
| Bonus `0.2 * answer_reward * Gate-I * Gate-II` | `compute_cagro_gate` |
| Bonus modifies the answer stream rather than adding a fifth stream | CAGRO reward assembly in `grpo_trainer.py` |
| Independent group normalization, then weights `(0.2, 0.7, 0.2, 0.2)` | Four calls to `compute_advantage`, weights applied afterward |
| Format to all generated tokens; context/think/answer to their own bodies | Token-level advantage assembly in `compute_loss` |
| `G=4`, `epsilon=0.2`, LR `5e-7`, accumulation `8`, max generation `1024`, ZeRO-2 | Config defaults and canonical stage-2 launcher |

## Explicit implementation choices where the paper is silent

- Group standard deviation uses population variance (`unbiased=False`) with
  `1e-4` numerical epsilon.
- Token losses are averaged within each completion and then across completions,
  preventing longer responses from receiving larger batch weight.
- The canonical launcher uses `beta=0`, temperature `0.7`, and top-p `0.9`.
  These values are launcher assumptions, not reported CAGRO hyperparameters.
- The launcher uses a remote fixed evaluator interface because no evaluator
  checkpoint/version is identified in the manuscript. Decoding is deterministic,
  but exact reproducibility still requires pinning the service model externally.
- Constrained forcing of `<context>` was removed: it was not described in the
  paper and would have optimized a token that the behavior policy could not
  freely sample. The model is instead stabilized through cold-start SFT and the
  format signal.


