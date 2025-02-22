# -*- coding: utf-8 -*-
# dsl_gen/backend/llm_judge.py
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langgraph.graph import END
from .client import getClient
from ..config import CFG
from ..core.rag import RAGState
import logging
from typing import Optional

logger = logging.getLogger('dsl_gen')


def _build_judge_prompt(question: str,
                        completion: str,
                        is_code: bool,
                        ref: Optional[str] = None,
                        ground_truth: Optional[str] = None) -> list:
    """Structured scoring prompt template"""

    # Dynamically construct problem type description
    problem_type_desc = (
        "Code logic problem" if is_code else "Question answering problem",
        "(pay special attention to code logic)" if is_code else ""
    )

    ref_line = "## References Provided\n{ref}\n\n" if ref else ""

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(CFG.JUDGE.personality),
        HumanMessagePromptTemplate.from_template(
            "## Problem Type\n{problem_type}\n\n"
            "## Original Question\n{question}\n\n"
            "## Answer\n```envision\n{completion}\n```\n\n"
            f"{ref_line}"
            "## Additional Reference (if provided)\n{ground_truth}\n\n"
            "Please analyze step by step and output the final judgment (0/1):"
        )
    ])

    return prompt_template.format_messages(
        problem_type=f"{problem_type_desc[0]}{problem_type_desc[1]}",
        question=question,
        completion=completion,
        ref=ref,
        ground_truth=ground_truth or "No additional reference materials"
    )


def judge_answer(state: RAGState) -> RAGState:
    """Node: LLM as judge
    ### Input fields
        question (str)
        completion (str): The answer or the successfully compiled code
        question_type (str): Either 'QA' or 'coding'
        (Optional) For evaluation mode (when a challenge path is provided):
            ground_truth (str)
            ref (str)
    ### Fields modified
        judgement (str): 'correct', 'incorrect' or 'internal judgment error'
        judge_output (str): The raw output from the judge
        judgement_attempts (int): Number of compilation attempts.
        final_state (str): 'success', 'compilation error' or 'judgment error'
    """
    judgement_attempts = state.get("judge_attempts", 0) + 1

    try:
        # Generate structured prompt
        messages = _build_judge_prompt(
            state["question"],
            state["completion"],
            state["question_type"] == "coding",
            ref=state.get("ref", None),
            ground_truth=state.get("ground_truth", None)
        )

        # Call LLM and parse response
        client = getClient(CFG.JUDGE.active_model, CFG.JUDGE.temperature)
        response = client.invoke(messages)

        logger.info("Judgment: %s", response.content)

        # Extract final judgment from response
        last_line = response.content.strip().split('\n')[-1].lower()
        if ('incorrect' in last_line) or ('0' in last_line):
            judgment = 'incorrect'
        elif ('correct' in last_line) or ('1' in last_line):
            judgment = 'correct'
        else:
            raise ValueError(f"Unexpected judgment result: {last_line}")

        state.update({
            "judge_attempts": judgement_attempts,
            "judge_output": response.content,
            "judgment": judgment,
            "final_state": "success" if judgment == "correct" else "judgment error"
        })

        logger.info(f"Judgment: {judgment}")
        return state

    except Exception as e:
        logger.error(f"Judgment failed (Internal error): {str(e)}")
        state.update({
            "judge_attempts": judgement_attempts,
            "judge_output": response.content,
            "judgment": "internal judgment error",
            "final_state": "judgment error"
        })
        return state


__all__ = ["judge_answer", "handle_judgment"]
