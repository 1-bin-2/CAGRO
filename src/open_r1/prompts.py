"""Shared response protocol used by both cold-start SFT and CAGRO."""

CAGRO_SYSTEM_PROMPT = """Use exactly these three tags in this order and do not output anything else:
<context>...</context>
<think>...</think>
<answer>...</answer>

Replace ... with concrete visual/audio evidence, concise reasoning, and the final answer requested by the question. Close context with </context>, close reasoning with </think>, and keep the answer payload concise. For multiple-choice questions use only option letter(s), such as A or B,E."""
