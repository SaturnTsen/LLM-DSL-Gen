# -*- coding: utf-8 -*-
# dsl_gen/agents/api.py

from ..config import API
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

import logging

logger = logging.getLogger('dsl_gen')

_llm_clients = {
    'openai': None,
    'ollama': None,
    'deepseek': None,
    'mistralai': None,
}

# For MistralAI version, replace every "open" by "mistral" in the following code

def getClient(backend) -> ChatOpenAI:

    global _llm_clients

    assert backend in _llm_clients, f"Invalid backend: {backend}"

    if _llm_clients[backend] is None:
        cfg = {
            'openai': API.MODEL_CFG.OpenAI,
            'ollama': API.MODEL_CFG.Ollama,
            'deepseek': API.MODEL_CFG.DeepSeek,
            'mistralai': API.MODEL_CFG.MistralAI
        }[backend]

        _llm_clients[backend] = ChatOpenAI(
            model=cfg.model,
            openai_api_base=cfg.base_url,
            openai_api_key=cfg.api_key
        )

        logger.info("Model loaded: %s, base_url: %s", cfg.model, cfg.base_url)

    return _llm_clients[backend]
