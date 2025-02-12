# -* coding: utf-8 *-
# dsl_gen/core/rag.py


import logging
from ..config import CFG
from langchain_core.documents import Document
from typing import List
from .state import RAGState
from .vector_store import get_vectorstore

logger = logging.getLogger('dsl_gen')

# Retrieve Docs Node

def _retrieve_docs(query: str, top_k: int = None,preprocessmethod="plain") -> List[Document]:
    """Enhanced retrieval function"""

    top_k = top_k or CFG.EMBEDDING_CFG.top_k

    try:
        vectorstore = get_vectorstore(preprocessmethod)
        # Example of a hybrid retrieval strategy: filtering based on metadata
        # TODO: Implement a more sophisticated filtering strategy
        # TODO: Add references to the RAG flow for better completion results
        return vectorstore.similarity_search(
            query,
            k=top_k
            # ,filter={
            #     "source": "official_docs"  # Assuming documents have a source metadata field
            # }
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}")
        return []


def retrieve_docs(state: RAGState) -> RAGState:
    """Document retrieval node
    ### Input fields
        question (str): The question to be answered.
        question_type (str): The type of question (QA or coding) # Not really used
    ### Fields added
        docs (List[Document]): The retrieved documents.
    """
    # Input fields: question, question_type
    # Output fields: docs

    # Defensive programming is IMPORTANT!!!
    assert "question" in state, "Question must be provided in the state"

    query = state["question"]
    logger.debug(f"Retrieving docs for query: {query}")
    if CFG.EMBEDDING_CFG.embedding_method == "plain":
        docs = _retrieve_docs(query)
    elif CFG.EMBEDDING_CFG.embedding_method == "pack":
        docs = _retrieve_docs(query,preprocessmethod="pack")
    else:
        raise ValueError(f"Unsupported packing method: {CFG.EMBEDDING_CFG.embedding_method}")
    logger.info(f"Retrieved {len(docs)} docs")
    state.update({"docs": docs})
    return state


__all__ = ["retrieve_docs"]
