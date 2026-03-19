"""
QY Domain Resource Server — Typos, Connections, and Plot Unscrambling.

Verifies model outputs for three QY sub-domains:
  - typos: Spelling correction (substring match)
  - connections: Word grouping puzzles (group accuracy)
  - unscrambling: Plot sentence reordering (Levenshtein-based ordering score)

The sub-domain is determined by the `data_source` field in the verify request body.
Each sub-domain uses the `label` field from extra_info as ground truth.
"""

import json
import logging
import re
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

logger = logging.getLogger(__name__)


# ─── Scoring Logic (adapted from legacy mjnemogym/qydomain/score.py) ─────────


def levenshtein_distance(A, B) -> int:
    """Levenshtein distance between two sequences (strings or lists)."""
    N, M = len(A), len(B)
    dp = [[0 for _ in range(M + 1)] for _ in range(N + 1)]
    for j in range(M + 1):
        dp[0][j] = j
    for i in range(N + 1):
        dp[i][0] = i
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[N][M]


def extract_answer(llm_answer: str) -> str:
    pattern = r'.* --- (.*?) --- .*'
    match = re.search(pattern, llm_answer)
    return match.group(1) if match else llm_answer


def extract_plot_summary(text: str) -> str:
    pattern = r'<PLOT_SUMMARY>(.*)</PLOT_SUMMARY>'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        pattern = r'<PLOT_SUMMARY>(.*)'
        match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else text


def score_unscrambling(ground_truth: str, llm_answer: str) -> float:
    llm_answer = extract_plot_summary(llm_answer)
    gt_sentences = [s.strip() for s in ground_truth.split('.')]
    ans_sentences = [
        s.strip() for s in llm_answer.split('.')
        if s.strip() not in ('</PLOT_SUMMARY>', '**End of Plot Summary**')
    ]
    gt_sentences = [s for s in gt_sentences if s]
    ans_sentences = [s for s in ans_sentences if s]

    ans_ordering = []
    for x in gt_sentences:
        if not ans_sentences:
            break
        best_match = None
        min_dist = float('inf')
        for candidate in ans_sentences:
            dist = levenshtein_distance(x, candidate)
            if dist < min_dist:
                min_dist = dist
                best_match = candidate
        if best_match:
            try:
                ans_ordering.append(ans_sentences.index(best_match))
            except ValueError:
                pass

    n = len(gt_sentences)
    if n == 0:
        return 0.0
    raw_distance = levenshtein_distance(list(range(n)), ans_ordering)
    return max(0.0, 1 - (raw_distance / n))


def last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def remove_boxed(s: str):
    left = "\\boxed{"
    try:
        if s[:len(left)] == left and s[-1] == "}":
            return s[len(left):-1]
        return None
    except Exception:
        return None


def group_words(words: list):
    groups = [set()]
    words = [w.strip().lower() for w in words]
    for word in words:
        if len(groups[-1]) == 4:
            groups.append(set())
        groups[-1].add(word)
    return groups


def score_connections(ground_truth: str, llm_answer: str) -> float:
    solution_matches = re.findall(r'<solution>(.*?)</solution>', llm_answer)
    if not solution_matches:
        solution_matches = re.findall(r'<solution>(.*?)</solution>', llm_answer.replace('\n', ''))
    if not solution_matches:
        solution_matches = re.findall(r'</solution>(.*?)</solution>', llm_answer)

    ground_truth_words = ground_truth.split(',')

    if len(solution_matches) == 0 and '\\boxed' in llm_answer:
        boxed = last_boxed_only_string(llm_answer)
        if boxed:
            no_box = remove_boxed(boxed)
            if no_box:
                clean_text = no_box.replace('\\text{', '').replace('}', '').replace('\\', '')
                solution_matches = [clean_text]

    solution_matches = [match.replace('\n', '') for match in solution_matches]
    if len(solution_matches) == 0:
        return 0.0

    if len(solution_matches) > 1:
        all_words = []
        num_words = len(ground_truth_words)
        for match in solution_matches:
            all_words.extend(match.split(','))
        solution_words = all_words[-num_words:]
    else:
        solution_words = solution_matches[-1].split(',')

    llm_groups = group_words(solution_words)
    gt_groups = group_words(ground_truth_words)

    correct = sum(1 for g in llm_groups if g in gt_groups)
    if len(gt_groups) == 0:
        return 0.0
    return correct / len(gt_groups)


def score_typos(ground_truth: str, llm_answer: str) -> float:
    solution_matches = re.findall(r'<solution>(.*?)</solution>', llm_answer)
    if solution_matches:
        parsed = solution_matches[-1]
    else:
        parsed = llm_answer.replace('<solution>', '').replace('</solution>', '')
        parsed = extract_answer(parsed)
    parsed = ' '.join(filter(None, parsed.strip().split('\n')))
    return 1.0 if ground_truth in parsed else 0.0


# Map data_source to scorer
SCORERS = {
    "typos": score_typos,
    "connections": score_connections,
    "unscrambling": score_unscrambling,
}


# ─── Gym Resource Server ─────────────────────────────────────────────────────


class QYDomainConfig(BaseResourcesServerConfig):
    pass


class QYDomainVerifyRequest(BaseVerifyRequest):
    """Verify request with QY-domain metadata.

    The data_source field identifies the sub-domain (typos/connections/unscrambling).
    The label field contains the ground truth for scoring.
    """
    data_source: str = ""
    label: str = ""


class QYDomainVerifyResponse(BaseVerifyResponse):
    data_source: str = ""
    extracted_answer: Optional[str] = None


class QYDomainResourcesServer(SimpleResourcesServer):
    config: QYDomainConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: QYDomainVerifyRequest) -> QYDomainVerifyResponse:
        # Extract assistant text from response
        assistant_text = ""
        for output_item in body.response.output:
            if output_item.type != "message":
                continue
            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue
                assistant_text += content_item.text

        data_source = body.data_source
        label = body.label

        scorer = SCORERS.get(data_source)
        if scorer is None:
            logger.warning(f"Unknown QY data_source: {data_source}, returning 0.0")
            reward = 0.0
        else:
            try:
                reward = float(scorer(label, assistant_text))
            except Exception as e:
                logger.error(f"Error scoring {data_source}: {e}")
                reward = 0.0

        return QYDomainVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_answer=assistant_text[:500] if assistant_text else None,
        )


if __name__ == "__main__":
    QYDomainResourcesServer.run_webserver()
