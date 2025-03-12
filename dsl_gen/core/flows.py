# -* coding: utf-8 *-
# dsl_gen/core/flows.py
import json
from langgraph.graph import StateGraph, END
from .state import RAGState
from .rag import retrieve_docs
from dsl_gen.core.prompts import generate_prompt, handle_compilation_error, handle_judge_error
from ..config import CFG
from ..agents.llm_coder import llm_coder
from ..agents.lokad_compiler import compile_code
from ..agents.llm_judge import judge_answer
from .utils import text_display, pretty_display
from typing import Optional
import logging
import re


logger = logging.getLogger('dsl_gen')

# Before reading the code, I suggest you to read the definition of the RAGState
# class in dsl_gen/core/rag.py


def qa_splitter(state: RAGState,
                pretty_print: Optional[bool] = None) -> RAGState:
    """
    QA Splitter node that processes the input state and returns an updated
    state.

    ### Input fields
        Either:
          question (str): The question to be answered.
          question_type (str): The type of question (QA or coding)
        Or:
          challenge_path (str): The path to the challenge json file
          see benchmarks/challenges/ for the format of the challenge json file
    ### Fields added
        If challenge_path is provided:
            question (str): The question to be answered.
            question_type (str): The type of question.
            ground_truth (str): The ground truth answer.
            ref (str): Title of the related reference document.
            final_state (str): set to 'pending'.
        Else (if a question is provided):
            The state is returned as is.
    ### Raises
        AssertionError: If neither or both 'question' and 'challenge_path' are
        provided in the state.
    """
    # Input fields: question, challenge_path
    # Output fields: question, challenge_path, ground_truth, ref, question_type

    assert ("question" in state) != ("challenge_path" in state), \
        "Exactly one of 'question' or 'challenge_path' must be provided"

    pretty_print = pretty_print or CFG.PRETTY_PRINT

    if state.get("question"):
        assert "question_type" in state, \
            "Question type must be provided in eval mode"
        logger.debug("QA Splitter: question provided, eval mode")
    elif state.get("challenge_path"):
        logger.debug("QA Splitter: challenge path provided, path=%s",
                     state["challenge_path"])

        # read the challenge path json file
        with open(state["challenge_path"], encoding="utf-8") as f:
            challenge = json.load(f)
        state.update({"challenge_path": str(state["challenge_path"]),
                      "question": challenge["question"],
                     "ground_truth": challenge["answer"],
                      "ref": challenge["ref"],
                      "question_type": challenge["type"],
                      "final_state": "pending"})
    if logger.isEnabledFor(logging.INFO):
        if pretty_print:
            pretty_display(["**Question**", state["question"]])
        else:
            text_display(["**Question**", state["question"]])
    return state


def completion_peeler(state: RAGState,
                      pretty_print: Optional[bool] = None) -> RAGState:
    """Answer peeler node
    ### Input fields
        raw_completion (AIMessage): The completion from the coder.
        question_type (str): The type of question (QA or coding)
    ### Fields added
        completion (str): The extracted completion.
        Note: if the question type is 'QA', the peeler only removes the chain of thought.
    """
    assert ("raw_completion" in state) and (state["raw_completion"] is not None), \
        "State missing required key 'completion'"
    assert ("question_type" in state) and (state["question_type"] is not None), \
        "State missing required key 'question_type'"
    logger.debug("Completion Peeler: received coder's completion %s",
                 state["raw_completion"])

    pretty_print = pretty_print or CFG.PRETTY_PRINT
    raw_completion_content = state["raw_completion"].content
    # Remove <think>...</think> tags
    completion_content = re.sub(r'<think[\s\S]*?>[\s\S]*?</think>',
                                '', raw_completion_content, flags=re.DOTALL)

    if state["question_type"] == "QA":
        logger.debug("Completion Peeler: QA question, removed CoT")
        state.update({"completion": completion_content})

    elif state["question_type"] == "coding":
        # Match code blocks enclosed in triple backticks or ```envision...```
        # code blocks
        code_blocks = re.findall(
            r'```(?:envision)?\s*\n?(.*?)\n?```', completion_content, flags=re.DOTALL)
        if not code_blocks:
            logger.warning(
                "No valid code blocks found in completion for a coding question")
            state.update({"completion": completion_content})
        else:
            extracted_code = code_blocks[0].strip()
            state.update({"completion": extracted_code})
    else:
        raise ValueError(
            f"Question type must be 'QA' or 'coding', got {state['question_type']}")
    if logger.isEnabledFor(logging.INFO):
        if pretty_print:
            pretty_display(["**Coder's Completion**",
                           f"```envision\n{state['completion']}```"])
        else:
            text_display(["**Coder's Completion**", state["completion"]])
    return state


def handle_judgment(state: RAGState) -> str:
    """Enhanced judgment logic
    ### Input fields
        judgment (str)
        judge_output (str)
    ### Fields modified
        judgement_attempts (int): Number of compilation attempts.
        final_state (str): when END is reached
    """
    assert "judgment" in state, "State missing required key 'judgment'"
    assert "judge_attempts" in state, "State missing required key 'judge_attempts'"
    assert state["judgment"] in ['correct', 'incorrect', 'internal judgment error'], \
        "Judgment must be one of 'correct', 'incorrect', or 'internal judgment error'"
    # Success condition
    if state["judgment"] == "correct":
        logger.info("Answer passed quality check")
        state["final_state"] = "success"
        return END
    # Max hit
    if state["judge_attempts"] >= CFG.JUDGE.max_retries:
        logger.warning("Reached maximum review attempts, forcing output")
        state["final_state"] = "judgment error"
        return END
    # Success condition
    if state["judgment"] == "incorrect":
        logger.info(
            f"Review attempt {state.get('judge_attempts', 0)} failed, regenerating answer")
        return "judge_error_handler"
    if state["judgment"] == "internal judgment error":
        return "judge"


def handle_compilation(state: RAGState) -> str:
    max_retries = CFG.COMPILER.max_retries
    # if compilation is successful, move to the judge
    if state["question_type"] == "QA" or state["compilation_result"]["valid"]:
        return "judge"
    elif state["compilation_attempts"] >= max_retries:
        logger.info("Max retires hit on compilation: %s", max_retries)
        state["final_state"] = "compilation error"
        return END
    else:
        return "compilation_error_handler"


def build_rag_flow():
    workflow = StateGraph(RAGState)
    # Preprocessor
    workflow.add_node("QA_splitter", qa_splitter)
    # RAG
    workflow.add_node("retriever", retrieve_docs)
    workflow.add_node("prompt_generator", generate_prompt)
    # Coder

    workflow.add_node("llm_coder", llm_coder)
    workflow.add_node("completion_peeler", completion_peeler)
    # Compiler
    workflow.add_node("compiler", compile_code)
    # Judge
    workflow.add_node("judge", judge_answer)
    workflow.add_node("judge_error_handler", handle_judge_error)
    workflow.add_node("compilation_error_handler", handle_compilation_error)

    workflow.set_entry_point("QA_splitter")
    workflow.add_edge("QA_splitter", "retriever")

    workflow.add_edge("retriever", "prompt_generator")
    workflow.add_edge("prompt_generator", "llm_coder")
    workflow.add_edge("llm_coder", "completion_peeler")
    workflow.add_edge("completion_peeler", "compiler")
    workflow.add_edge("judge_error_handler", "llm_coder")

    workflow.add_edge("compilation_error_handler", "llm_coder")
    workflow.add_conditional_edges(
        "compiler",
        handle_compilation,
        {
            "judge": "judge",
            "compilation_error_handler": "compilation_error_handler",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "judge",
        handle_judgment,
        {
            "judge_error_handler": "judge_error_handler",
            "judge": "judge",
            END: END
        }
    )
    return workflow.compile()
