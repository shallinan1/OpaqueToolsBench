"""Lightweight evaluation helpers for BrowseCompPlus.

This module mirrors the judge prompt/parse logic from the vendor evaluator,
but avoids importing vLLM-heavy dependencies at import time.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. 

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def load_ground_truth(jsonl_path: Path) -> Dict[str, Dict[str, str]]:
    gt: Dict[str, Dict[str, str]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            obj = json.loads(line)
            gt[str(obj["query_id"])] = {
                "question": obj["query"],
                "answer": obj["answer"],
            }
    return gt


def create_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    return GRADER_TEMPLATE.format(
        question=question,
        response=response,
        correct_answer=correct_answer,
    )


def parse_judge_response(judge_response: str) -> dict:
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }

    if not judge_response:
        result["parse_error"] = True
        return result

    answer_match = re.search(r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if not answer_match:
        answer_match = re.search(r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if not answer_match:
        answer_match = re.search(r"extracted_final_answer:\s*(.*?)(?=\n|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    reasoning_match = re.search(r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if not reasoning_match:
        reasoning_match = re.search(r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if not reasoning_match:
        reasoning_match = re.search(r"reasoning:\s*(.*?)(?=\ncorrect:|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    correct_match = re.search(r"\*\*correct:\*\*\s*(yes|no)", judge_response, re.IGNORECASE)
    if not correct_match:
        correct_match = re.search(r"\*\*correct\*\*:\s*(yes|no)", judge_response, re.IGNORECASE)
    if not correct_match:
        correct_match = re.search(r"correct:\s*(yes|no)", judge_response, re.IGNORECASE)
    if correct_match:
        result["correct"] = correct_match.group(1).lower() == "yes"

    confidence_match = re.search(r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if not confidence_match:
        confidence_match = re.search(r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if not confidence_match:
        confidence_match = re.search(r"confidence:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if confidence_match:
        result["confidence"] = float(confidence_match.group(1))
        if result["confidence"] > 100:
            result["confidence"] = 100

    if result["correct"] is None:
        result["parse_error"] = True

    return result


def extract_citations_from_response(response_text: str) -> List[str]:
    """Extract citations from response text in [] and 【】 forms."""
    if not response_text:
        return []

    single_citation_pattern = r"\[(\d+)\]"
    single_matches = re.findall(single_citation_pattern, response_text)

    multi_citation_pattern = r"\[([^\[\]]*?)\]"
    multi_matches = re.findall(multi_citation_pattern, response_text)

    single_fullwidth_pattern = r"【(\d+)】"
    single_fullwidth_matches = re.findall(single_fullwidth_pattern, response_text)

    multi_fullwidth_pattern = r"【([^【】]*?)】"
    multi_fullwidth_matches = re.findall(multi_fullwidth_pattern, response_text)

    all_docids = set()
    all_docids.update(single_matches)
    all_docids.update(single_fullwidth_matches)

    for match in multi_matches:
        if match in single_matches:
            continue
        docids = re.findall(r"\d+", match)
        all_docids.update(docids)

    for match in multi_fullwidth_matches:
        if match in single_fullwidth_matches:
            continue
        docids = re.findall(r"\d+", match)
        all_docids.update(docids)

    return list(all_docids)


def load_qrel_data(qrel_path: Path) -> Dict[str, List[str]]:
    qrel_data = defaultdict(list)

    if not qrel_path.exists():
        return dict(qrel_data)

    with qrel_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            assert len(parts) == 4, f"Expected 4 parts in line: {line}"
            query_id = parts[0]
            doc_id = parts[2]
            qrel_data[query_id].append(doc_id)

    return dict(qrel_data)


def compute_citation_metrics(cited_docids: List[str], relevant_docids: List[str]) -> Dict[str, float]:
    metrics = {
        "num_citations": len(cited_docids),
        "num_relevant": len(relevant_docids),
        "precision": 0.0,
        "recall": 0.0,
    }

    if len(cited_docids) == 0:
        return metrics

    cited_set = set(cited_docids)
    relevant_set = set(relevant_docids)
    relevant_cited = cited_set & relevant_set

    if len(cited_docids) > 0:
        metrics["precision"] = len(relevant_cited) / len(cited_docids)
    if len(relevant_docids) > 0:
        metrics["recall"] = len(relevant_cited) / len(relevant_docids)

    return metrics
