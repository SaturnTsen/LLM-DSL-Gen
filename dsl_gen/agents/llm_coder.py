# -*- coding: utf-8 -*-
# dsl_gen/agents/llm_coder.py

from .client import getClient
from ..core.rag import RAGState
from ..config import CFG
from langchain_core.messages import BaseMessage, AIMessage
from typing import List
import logging

logger = logging.getLogger('dsl_gen')


def _generate_answer(messages: List[BaseMessage]) -> AIMessage:
    client = getClient(CFG.CODER.active_model, CFG.CODER.temperature)
    response = client.invoke(messages)
    logger.debug("Model's response:\n %s\n", response.model_dump()["content"])
    return response


def llm_coder(state: RAGState) -> RAGState:
    """LLM Coder node
    ### Input fields
        messages (str): The formatted prompt messages, including the question,
        question type, and the retrieved documents.
    ### Fields added
        raw_completion (str): The generated raw completion.
    """
    assert "messages" in state, "messages field is required"
    assert isinstance(state["messages"], list), "messages field must be a list"
    assert all(isinstance(m, BaseMessage) for m in state["messages"]), \
        "messages field must be a list of BaseMessage instances"

    state.update({"raw_completion": _generate_answer(state["messages"])})
    return state
