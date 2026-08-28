from transformers import Qwen2_5OmniThinkerForConditionalGeneration, AutoProcessor, Qwen2_5OmniProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from qwen_omni_utils import process_mm_info

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import os
import re
import ast
import time
import math
from open_r1.vlm_modules.vlm_module import VLMBaseModule
from open_r1.cagro import extract_unique_tag_text, parse_strict_cagro_response
import requests


USE_API_REWARD = os.environ.get("USE_API_REWARD", "0") == "1"

url = ""
token = ""

if USE_API_REWARD:
    url = os.environ.get("API", "").rstrip("/")
    if not url:
        raise RuntimeError("USE_API_REWARD=1, but API is not set.")

    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"

    token = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not token:
        raise RuntimeError("USE_API_REWARD=1, but API key is not set.")

def gpt_api(prompt, model_name="qwen-plus"):
    if not USE_API_REWARD:
        return "0"
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    max_try = 2
    last_error = None

    for tries in range(max_try):
        try:
            data = {
                "model": model_name or "qwen-plus",
                "messages": messages,
                # The paper requires frozen evaluators with fixed decoding.
                # Greedy score generation removes provider-default sampling
                # variance from context/reasoning supervision.
                "temperature": 0.0,
                "max_tokens": 16,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + token,
            }

            # print(f"[GPT_API] try {tries + 1}/{max_try}, model={data['model']}", flush=True)

            response = requests.post(
                url,
                json=data,
                headers=headers,
                timeout=20,
            )

            response.raise_for_status()
            response_json = response.json()
            response_message = response_json["choices"][0]["message"]["content"].strip()

            # print("[GPT_API] success", flush=True)
            return response_message

        except Exception as e:
            last_error = e
            body = ""
            try:
                body = response.text[:300]
            except Exception:
                pass

            print(f"[GPT_API_ERROR] try {tries + 1}/{max_try}: {type(e).__name__}: {e}", flush=True)
            if body:
                print(f"[GPT_API_BODY] {body}", flush=True)

            time.sleep(0.5)

    print(f"[GPT_API_FAILED] return 0. last_error={last_error}", flush=True)
    return "0"


def _api_score_0_to_5(raw_score):
    """Parse a model judge score without silently discarding valid responses."""
    try:
        score = float(ast.literal_eval(str(raw_score).strip()))
    except Exception:
        match = re.search(r"(?<!\d)([0-5](?:\.\d+)?)(?!\d)", str(raw_score))
        if not match:
            raise ValueError(f"API reward did not contain a numeric 0-5 score: {raw_score!r}")
        score = float(match.group(1))
    return max(0.0, min(5.0, score)) / 5.0


def _map_api_rewards(items, reward_fn, question_type):
    """Evaluate API-backed rewards concurrently while preserving input order."""
    try:
        max_workers = max(1, int(os.environ.get("API_REWARD_MAX_WORKERS", "4")))
    except (TypeError, ValueError):
        max_workers = 4

    def safe_reward(item):
        try:
            return reward_fn(*item)
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            return 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(safe_reward, items))


class QwenOmniModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        
        return Qwen2_5OmniThinkerForConditionalGeneration
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return Qwen2_5OmniProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual','audio_tower']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'video_second_per_grid', 'feature_attention_mask', 'input_features', 'audio_feature_lengths', 'use_audio_in_video', 'rope_deltas']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return ['max_pixels', 'min_pixels']
    
    def prepare_prompt(self, processing_class, inputs):
        # The SFT checkpoint was trained with the processor's Qwen chat
        # template.  Preserve system/user role markers and append the assistant
        # generation marker here as well; flattening the conversation to raw
        # text causes literal "assistant" text and prompt leakage in answers.
        prompts = []
        for example in inputs:
            rendered = maybe_apply_chat_template(example, processing_class)["prompt"]
            # Qwen2.5-OmniProcessor returns a one-element list for one
            # multimodal conversation, while the downstream processor expects
            # each batch entry to be a plain string.
            if isinstance(rendered, list):
                if len(rendered) != 1 or not isinstance(rendered[0], str):
                    raise TypeError(
                        "Qwen2.5-Omni chat template must return exactly one string per prompt"
                    )
                rendered = rendered[0]
            if not isinstance(rendered, str):
                raise TypeError(f"Unexpected chat-template result: {type(rendered)!r}")
            prompts.append(rendered)
        return prompts
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, audios, videos, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False, use_audio_in_video=False):

        # print(audios)
        video_token = getattr(processing_class, "video_token", "<tool_call>")
        video_token_id = processing_class.tokenizer.convert_tokens_to_ids(video_token)

        # print("[DEBUG_VIDEO_TOKEN]", video_token, video_token_id, flush=True)
        prompt_inputs = processing_class(
            text=prompts_text,
            images=images,
            audio=audios,
            videos=videos,
            return_tensors=return_tensors,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
            use_audio_in_video=use_audio_in_video)
        if "input_features" in prompt_inputs and "feature_attention_mask" in prompt_inputs:
            feature_len = prompt_inputs["input_features"].shape[-1]
            mask_len = prompt_inputs["feature_attention_mask"].shape[-1]
            if feature_len != mask_len:
                common_len = min(feature_len, mask_len)
                prompt_inputs["input_features"] = prompt_inputs["input_features"][..., :common_len]
                prompt_inputs["feature_attention_mask"] = prompt_inputs["feature_attention_mask"][..., :common_len]
       

       
        video_token = getattr(processing_class, "video_token", "<|video_pad|>")
        video_token_id = processing_class.tokenizer.convert_tokens_to_ids(video_token)

        # if videos is not None:
        #     print(
        #         "[DEBUG_VIDEO_TOKEN_COUNT_IN_IDS]",
        #         (prompt_inputs["input_ids"] == video_token_id).sum().item(),
        #         flush=True,
        #     )
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        # match task_type:
        #     case "rec":
        #         return "{Question}\nOutput exactly: <context>...</context><think>...</think><answer>...</answer>. Answer must be JSON."
        #     case _:
        #         return "{Question}\nOutput exactly: <context>...</context><think>...</think><answer>...</answer>. Answer only the option letter."
        # return (
        #     "{Question}\n\n"
        #     "Use exactly these three tags in this order and do not output anything else:\n"
        #     "<context>...</context><think>...</think><answer>...</answer>\n"
        #     "Replace ... with your own evidence, reasoning, and final option letter(s). "
        #     "The answer tag must contain only option letter(s), such as A or B,E."
        # )
        return "{Question}"

    @staticmethod
    def _extract_tag_answer(text):
        return extract_unique_tag_text(str(text), "answer")

  
    @staticmethod
    def _canonical_choice_answer(text, options=None):
        """Safely parse option letters or exact option text."""
        raw = re.sub(r"\s+", " ", str(text)).strip()
        if not raw:
            return ()

        raw = re.sub(
            r"^assistant\b[\s:,-]*",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip()

        option_entries = []

        for index, option in enumerate(options or []):
            match = re.match(
                r"^\s*([A-J])\s*[.\):：-]\s*(.*?)\s*$",
                str(option),
                re.DOTALL | re.IGNORECASE,
            )

            if match:
                label = match.group(1).upper()
                value = match.group(2).strip()
            else:
                label = chr(ord("A") + index)
                value = str(option).strip()

            option_entries.append((label, value))

        valid_labels = {label for label, _ in option_entries}

        # 支持 A、B,E、The answer is B，但不接受 “The answer is a cat”
        # 被无条件误判为 A。
        explicit = re.sub(
            r"^(?:the\s+)?(?:final\s+)?"
            r"(?:answer|option(?:s)?)\s*"
            r"(?:is|are|:)?\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip()

        explicit = explicit.rstrip(".。;；!！")

        if re.fullmatch(
            r"[A-J](?:\s*[,，/]\s*[A-J])*",
            explicit,
            re.IGNORECASE,
        ):
            labels = tuple(
                sorted(set(re.findall(r"[A-J]", explicit.upper())))
            )

            if labels and all(label in valid_labels for label in labels):
                return labels

            return ()

        def normalize_option_text(value):
            value = str(value).casefold()
            value = re.sub(
                r"[^\w.%+-]+",
                " ",
                value,
                flags=re.UNICODE,
            )
            value = re.sub(r"\s+", " ", value).strip()
            value = re.sub(r"^(?:the|a|an)\s+", "", value)
            value = re.sub(
                r"\s+(?:model|option|choice)$",
                "",
                value,
            )
            return value

        # 支持 <answer>Yes</answer> 这种直接输出选项内容的情况，
        # 但必须与唯一一个选项精确对应。
        normalized_answer = normalize_option_text(explicit)
        matches = []

        for label, value in option_entries:
            if value and normalized_answer == normalize_option_text(value):
                matches.append(label)

        unique_matches = sorted(set(matches))

        if len(unique_matches) == 1:
            return (unique_matches[0],)

        return ()
            
    @staticmethod
    def _has_complete_reasoning_answer_format(text):
        return parse_strict_cagro_response(str(text)) is not None

    @staticmethod
    def _has_parseable_reasoning_answer_format(text):
        # answer_re = r"[A-Z](?:\s*,\s*[A-Z])*"
        # pattern = (
        #     rf"^\s*<context>\s*[^<>]+?\s*</context>\s*"
        #     rf"<think>\s*[^<>]+?\s*</think>\s*"
        #     rf"<answer>\s*{answer_re}\s*</answer>"
        # )
        # return re.search(pattern, str(text).strip(), re.DOTALL) is not None
        return QwenOmniModule._has_complete_reasoning_answer_format(text)
    
    @staticmethod
    def format_reward(completions, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        placeholder_values = {
            "answer",
            "answers",
            "context",
            "evidence",
            "reason",
            "think",
            "thinking",
            "brief evidence",
            "brief reason",
            "visual/audio evidence",
            "reasoning",
            "option letter(s)",
            "...",
            "[...]",
        }

        def is_placeholder(value):
            normalized = re.sub(r"\s+", " ", value).strip().lower()
            return normalized in placeholder_values

        rewards = []
        for text in contents:
            parsed = parse_strict_cagro_response(str(text))
            leaked_chat_token = re.search(
                r"<\|(?:im_start|im_end|eot_id|endoftext)\|>",
                str(text),
                re.IGNORECASE,
            )
            if parsed is None or leaked_chat_token:
                rewards.append(0.0)
                continue

            bodies = [
                str(text)[parsed[tag].start:parsed[tag].end]
                for tag in ("context", "think", "answer")
            ]
            rewards.append(0.0 if any(is_placeholder(body) for body in bodies) else 1.0)

        return rewards
    
    @staticmethod
    def _legacy_reasoning_answer_consistency_reward(completions, solution, **kwargs):
        """Deprecated legacy reward retained only for checkpoint archaeology.

        This signal is intentionally private and is not registered by CAGRO: the
        paper specifies exactly four base signals and no fifth consistency reward.
        """

        stop_words = {
              "this", "that", "with", "from", "they", "them", "their", "there",
              "have", "has", "had", "were", "was", "will", "would", "could",
              "should", "about", "into", "onto", "when", "where", "which",
              "what", "while", "because", "then", "than", "also", "very",
              "more", "most", "some", "such", "only", "just", "being", "been",
              "does", "doing", "done", "the", "and", "for", "are", "but",
              "not", "you", "your", "his", "her", "its", "our", "out",
              "all", "can",
          }

        def extract_part(text, tag):
            pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        def content_terms(text):
            return {
                word
                for word in re.findall(r"[A-Za-z]{4,}", text.lower())
                if word not in stop_words
            }

        def has_repetition(text):
            sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
            seen = set()

            for sentence in sentences:
                normalized = re.sub(
                    r"\s+", " ", sentence.strip().lower()
                )

                if len(normalized) < 18:
                    continue

                if normalized in seen:
                    return True

                seen.add(normalized)

            return False

        def build_option_map(options):
            option_map = {}

            for index, option in enumerate(options or []):
                option = str(option).strip()

                match = re.match(
                    r"^\s*([A-Z])[\.\)\:\-]\s*(.*)$",
                    option,
                    re.IGNORECASE,
                )

                if match:
                    label = match.group(1).upper()
                    option_text = match.group(2).strip()
                else:
                    label = chr(ord("A") + index)
                    option_text = option

                option_map[label] = option_text

            return option_map

        # Reuse the exact Accuracy Reward implementation as the correctness gate.
        accuracy_scores = QwenOmniModule.accuracy_reward(
            completions,
            solution,
            **kwargs,
        )

        completion_contents = [
            completion[0]["content"] for completion in completions
        ]
        question_types = kwargs.get(
            "problem_type",
            [""] * len(completions),
        )
        option_lists = kwargs.get(
            "options",
            [[] for _ in completions],
        )

        rewards = []

        for (
            content,
            sol,
            question_type,
            options,
            accuracy_score,
        ) in zip(
            completion_contents,
            solution,
            question_types,
            option_lists,
            accuracy_scores,
        ):
            context = extract_part(content, "context")
            think = extract_part(content, "think")
            raw_answer = extract_part(content, "answer")

            accuracy_score = float(accuracy_score or 0.0)

            # Critical safety gate:
            # malformed or incorrect answers never receive consistency reward.
            if (
                accuracy_score <= 0.0
                or not context
                or not think
                or not raw_answer
            ):
                rewards.append(0.0)
                continue

            context_terms = content_terms(context)
            think_terms = content_terms(think)

            # Evidence consistency: does reasoning use information from context?
            if context_terms and think_terms:
                evidence_overlap = len(
                    context_terms & think_terms
                ) / max(
                    1,
                    min(len(context_terms), len(think_terms)),
                )
            else:
                evidence_overlap = 0.0

            # 25% lexical overlap is treated as full evidence support.
            evidence_support = min(
                1.0,
                evidence_overlap / 0.25,
            )

            answer_support = 0.0

            if question_type in ("multiple choice", "emer_ov_mc"):
                selected_labels = (
                    QwenOmniModule._canonical_choice_answer(
                        raw_answer,
                        options,
                    )
                )

                option_map = build_option_map(options)
                selected_option_terms = set()

                for label in selected_labels:
                    selected_option_terms |= content_terms(
                        option_map.get(label, "")
                    )

                # Prefer support for the semantic content of the option.
                if selected_option_terms and think_terms:
                    support_overlap = len(
                        selected_option_terms & think_terms
                    ) / max(
                        1,
                        min(
                            len(selected_option_terms),
                            len(think_terms),
                        ),
                    )

                    answer_support = min(
                        1.0,
                        support_overlap / 0.25,
                    )

                # "Option A" is valid explicit support.
                # A standalone English article "a" is not.
                for label in selected_labels:
                    if re.search(
                        rf"\bOPTION\s+{re.escape(label)}\b",
                        think,
                        re.IGNORECASE,
                    ):
                        answer_support = max(answer_support, 0.25)
                        break

            else:
                normalized_answer = raw_answer.strip()

                if normalized_answer:
                    if re.search(
                        rf"(?<!\w){re.escape(normalized_answer)}(?!\w)",
                        think,
                        re.IGNORECASE,
                    ):
                        answer_support = max(answer_support, 0.25)

            # Accuracy is only a gate/scale. The remaining score comes from
            # context-to-reasoning and reasoning-to-answer support.
            reward = accuracy_score * (
                0.55 * evidence_support
                + 0.45 * answer_support
            )

            bad_markers = [
                "<LM>",
                "</LM>",
                "Human:",
                "\nHuman",
                "<|im_start|>user",
            ]

            if any(marker in content for marker in bad_markers):
                reward -= 0.40

            if (
                content.count("<answer>") != 1
                or content.count("</answer>") != 1
            ):
                reward -= 0.25

            if has_repetition(think):
                reward -= 0.35

            rewards.append(
                max(0.0, min(1.0, reward))
            )

        return rewards
    

    @staticmethod
    def precision_reward(completions, solution, **kwargs):

        completions = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion, sol in zip(completions, solution):
            reward = 0.0
            # print(completion, sol)
            answer_tag_pattern = r'<answer>(.*?)</answer>'
            # Try symbolic verification first
            # try:
            content_answer_match = re.search(answer_tag_pattern, completion, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                words = content_answer.split(",")
                count = 0
                for each in sol:
                    if each.lower() in content_answer or each in content_answer:
                        count +=1

                reward = float(count)/len(sol)
                # bbox_match = re.search(bbox_pattern, content_answer)
            rewards.append(reward)
            # except Exception as e :
            #     pass  # Continue to next verification method if this fails
        # print(rewards)
        return rewards
      
        
    @staticmethod
    def recall_reward(completions, solution, **kwargs):
        import re
        completions = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion, sol in zip(completions, solution):
            reward = 0.0
            # print(completion, sol)
            answer_tag_pattern = r'<answer>(.*?)</answer>'
            # Try symbolic verification first
            # try:
            content_answer_match = re.search(answer_tag_pattern, completion, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                words = content_answer.split(",")
                count = 0
                for each in sol:
                    if each.lower() in content_answer or each in content_answer:
                        count +=1

                reward = float(count)/len(sol)
                # bbox_match = re.search(bbox_pattern, content_answer)
            rewards.append(reward)
            # except Exception as e :
            #     pass  # Continue to next verification method if this fails
        # print(rewards)
        return rewards

    @staticmethod
    def accuracy_reward(completions, solution, **kwargs):
    
        def extract_answer(text):
            return extract_unique_tag_text(str(text), "answer")
        
        def normalize_number(num_str):
            try:
                value = float(str(num_str).replace(",", "").strip())
            except (TypeError, ValueError):
                return None

            if not math.isfinite(value):
                return None

            return value

        def wer(reference, hypothesis):
            ref_words = reference.split()
            hyp_words = hypothesis.split()
            m = len(ref_words)
            n = len(hyp_words)
            d = [[0]*(n+1) for _ in range(m+1)]
            for i in range(m+1):
                d[i][0] = i
            for j in range(n+1):
                d[0][j] = j
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if ref_words[i-1] == hyp_words[j-1]:
                        d[i][j] = d[i-1][j-1]
                    else:
                        d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
            return d[m][n] / max(1, m)


        def compute_rouge_score(reference, hypothesis, use_stemmer=True):
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
            scores = scorer.score(reference, hypothesis)
            average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
            return average_fmeasure

        def similarity(reference, hypothesis):
            prompt = f"""
            Analyze the consistency between the content of the two compared texts and assign a score based on the following criteria:

            Grading criteria description (content consistency):

            5 points: The core facts, details, and logical relationships in the two texts are entirely consistent, with no differences.
            3-4 points: The core content is consistent, but there are differences in non-critical details (such as expression, supplementary information, examples, etc.).
            1-2 points: Some content is consistent, but there are contradictions or differences in key information.
            0 points: The core content is inconsistent or completely irrelevant.

            Example analysis process:

            Extract the core information from both texts (time, place, people, events, data, conclusions, etc.).
            Compare whether key facts align (e.g., whether the times are the same, whether the data matches).
            Analyze the consistency of logical relationships (causal relationships, sequence, etc.).
            Determine whether the differences are merely expressive (such as synonym replacement, sentence adjustment) or substantive content differences.

            reference: {reference}
            hypothesis: {hypothesis}

            only return the score number:
            """
    
            try:
                reward = gpt_api(prompt=prompt, model_name="qwen-plus")
                reward = _api_score_0_to_5(reward)
            except:
                return 0

            return reward
        
 
        
        def emer_ov_mc(reference, hypothesis):
            list_a = reference.split(",")
            list_b = hypothesis.split(",")
            true_positive = len(set(list_a) & set(list_b))
            precision = true_positive / len(list_a) if list_a else 0
            recall = true_positive / len(list_b) if list_b else 0
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0
            
            return f1_score
        
        def judge(reference, hypothesis):
            reference = re.sub(r"[^a-z]", "", str(reference).lower())
            hypothesis = re.sub(r"[^a-z]", "", str(hypothesis).lower())
            return int(reference in {"yes", "no"} and reference == hypothesis)


        # question_type = kwargs['problem_type'][0]

        question_types = kwargs.get("problem_type", [""] * len(completions))
        option_lists = kwargs.get("options", [[] for _ in completions])
        
        contents = [completion[0]["content"] for completion in completions]
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        rewards = []

        # for content, sol in zip(contents, solution):
        for content, sol, question_type, options in zip(contents, solution, question_types, option_lists):
            try:
                output_ans = extract_answer(content)
                gt_ans = extract_answer(sol)
                if question_type == "multiple choice":
                    output_letters = QwenOmniModule._canonical_choice_answer(output_ans, options)
                    gt_letters = QwenOmniModule._canonical_choice_answer(gt_ans, options)
                    reward = 1.0 if output_letters and output_letters == gt_letters else 0.0
                elif question_type == "numerical":
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
                elif question_type == "OCR":
                    error_rate = wer(gt_ans, output_ans)
                    reward = 1 - error_rate
                    reward = max(0.0, min(1.0, reward))
                elif question_type == "free-form":
                    # reward = similarity(gt_ans, output_ans)
                    score = compute_rouge_score(gt_ans, output_ans)
                    reward = max(0.0, min(1.0, score))
                elif question_type == "regression":
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        abs_error = abs(out_number - gt_number)

                        # Correctly handle target == 0.
                        if abs(gt_number) < 1e-8:
                            reward = 1.0 if abs_error <= 1e-6 else 0.0
                        else:
                            rel_diff = abs_error / abs(gt_number)
                            rel_diff = min(1.0, max(0.0, rel_diff))
                            reward = 1.0 - rel_diff
                        
                elif question_type == "emer_ov":
                    reward = emer_ov_mc(gt_ans, output_ans)
                elif question_type == "emer_ov_mc":
                    output_letters = QwenOmniModule._canonical_choice_answer(output_ans, options)
                    gt_letters = QwenOmniModule._canonical_choice_answer(gt_ans, options)
                    reward = 1.0 if output_letters and output_letters == gt_letters else 0.0
                    # reward = emer_ov_mc(gt_ans, output_ans)
                elif  question_type == "judge":
                    reward = judge(output_ans, gt_ans)
                else:
                    reward = 0.0
            except Exception as e:
                print(f"Error in reward_fn for question_type '{question_type}': {e}")
                reward = 0.0
        
            rewards.append(reward)
            
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
                
        return rewards

    @staticmethod
    def context_reward(completions, solution, **kwargs):
    
        def extract_parts(text, pattern):
            del pattern
            return extract_unique_tag_text(str(text), "context")

        def similarity(reference, hypothesis):

            prompt = \
f"""You are assessing how well the 'hypothesis' text covers the key information from the 'reference' text. Differences in wording or extra details in the 'hypothesis' are fine if the 'reference's' main points are included.:

Score based on this coverage:

5 points : Hypothesis clearly and accurately reflects significant core themes or key aspects of the reference. It demonstrates a good understanding of a substantial part of the reference material.
4 points : Hypothesis reflects some important themes or aspects of the reference. The connection is evident, though perhaps not as comprehensive or central as a 5.
2 points : Hypothesis shows a recognizable connection to themes or aspects of the reference, but it might be more superficial, focus on less central points, or only partially grasp a key aspect.
1 points : Hypothesis has a tenuous or very limited connection to the reference. It might touch on a peripheral detail or a heavily reinterpreted aspect, but largely misses the main substance.
0 points : Hypothesis does not reflect any significant themes or key aspects of the reference, or is on a completely different topic.

Example analysis process:

Identify main themes and key aspects in 'reference'.
Determine if 'hypothesis' connects to or discusses any of these themes/aspects from 'reference'.
Judge the strength and relevance of this connection. Is a core part of the 'reference' reflected?
Differences are expected; evaluate if the 'hypothesis' still meaningfully reflects some key part of the 'reference'.
Assign score based on how well a significant aspect is reflected.

reference: {reference}
hypothesis: {hypothesis}

only return the score number:"""
    
            try:
                reward = gpt_api(prompt=prompt, model_name="qwen-plus")
                reward = _api_score_0_to_5(reward)
            except:
                return 0

            return reward
           


        question_type = kwargs['problem_type'][0]
        
        contents = [completion[0]["content"] for completion in completions]

        def reward_one(content, sol):
            output_evidence = extract_parts(content, pattern=r'<context>\s*(.*?)\s*</context>')
            gt_evidence = extract_parts(sol, pattern=r'<context>\s*(.*?)\s*</context>')
            if len(gt_evidence) == 0:
                return 0.0
            return similarity(gt_evidence, output_evidence)

        rewards = _map_api_rewards(
            list(zip(contents, solution)),
            reward_one,
            question_type,
        )
            
     
        return rewards

    @staticmethod
    def reasoning_reward(completions, solution, **kwargs):
    
        def extract_parts(text, pattern):
            tag_match = re.search(r"<([A-Za-z][A-Za-z0-9_-]*)>", pattern)
            if tag_match is None:
                return ""
            return extract_unique_tag_text(str(text), tag_match.group(1))

        def rationality(reference, hypothesis):
            
           
            prompt = \
f"""Please analyze whether the reasoning text is derived from the evidence and context text based on the following criteria and give a score of 0-5:
Grading criteria description (relevance and rationality):

Integration of Clues (1 point): During the reasoning process, there is incorporation of clues from the video, image, or audio.

Reflection and Confirmation (1 point): The reasoning involves reflection or second confirmation of choices or answers, including revisiting video, image, or audio evidence.

Logical Reasoning (1 point): The thought process is clear, deriving conclusions through rigorous logical reasoning, analysis, or extension without additional assumptions or contradictions.

Problem Analysis (1 point): The reasoning process includes thorough analysis in conjunction with the problem at hand.

Overall Consistency (1 point): The reasoning text is based on visual or audio evidence and context information, presenting no extra assumptions or contradictions.

Assign one point for each criterion that is met, for a total possible score of five points. Verify that each criterion is addressed and reflect this in your scoring.

context: {reference}
reasoning path: {hypothesis}

only return the score number:
            """
            try:
                reward = gpt_api(prompt=prompt, model_name="qwen-plus")
                reward = _api_score_0_to_5(reward)
            except:
                return 0

            return reward


        question_type = kwargs['problem_type'][0]
        
        contents = [completion[0]["content"] for completion in completions]

        def reward_one(content, sol):
            evidence = extract_parts(content, pattern=r'<context>\s*(.*?)\s*</context>')
            think_path = extract_parts(content, pattern=r'<think>\s*(.*?)\s*</think>')
            answer = extract_parts(content, pattern=r'<answer>\s*(.*?)\s*</answer>')

            if len(evidence) == 0 or len(think_path) == 0:
                return 0.0
            # output_think = extract_parts(content, pattern=r'<think>\s*(.*?)\s*</think>')
            return rationality(evidence, think_path)

        rewards = _map_api_rewards(
            list(zip(contents, solution)),
            reward_one,
            question_type,
        )
            
     
        return rewards

        
