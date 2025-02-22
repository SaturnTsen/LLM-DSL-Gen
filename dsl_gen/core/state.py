from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_core.prompts import (ChatPromptTemplate,
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate)
from typing import Any, List, Optional, TypedDict, Literal, Dict
from langchain_core.documents import Document


class RAGState(TypedDict):
    # Input
    challenge_path: Optional[str]  # Path to the challenge json file
    question: Optional[str]  # Question to be answered
    question_type: Optional[Literal["QA", "coding"]]  # Question type

    # For evaluation mode
    ground_truth: Optional[str]  # Ground truth answer
    ref: Optional[str]  # Reference answer

    # Retrieved documents
    docs: Optional[List[Document]]

    # Input to the coder
    messages: Optional[List[BaseMessage]]

    # Output from the coder
    raw_completion: Optional[AIMessage]
    completion: Optional[str]  # The last completion

    # Output from the compiler
    # compilation_result = {
    #     "valid": bool,
    #     "messages": List[str]
    # }
    compilation_result: Optional[Dict[str, Any]]
    compilation_attempts: int

    # Output from the judge
    # Either 'correct', 'incorrect' or 'internal judgment error'
    judgment: Optional[str]
    judge_output: Optional[str]
    judge_attempts: int

    final_state: Optional[Literal['pending',
                                  'success',
                                  'compilation error',
                                  'judgment error']]


def RAGState_to_dict(state: RAGState) -> Dict[str, Any]:
    return {
        "challenge_path": state.get('challenge_path'),
        "question": state.get('question'),
        "question_type": state.get('question_type'),
        "ground_truth": state.get('ground_truth'),
        "ref": state.get('ref'),
        "docs": [{
            'id': doc.id,
            "page_content": doc.page_content,
            "metadata": doc.metadata
        } for doc in state.get('docs', [])],
        "messages": [{
            'type': message.type,
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
            "response_metadata": message.response_metadata
        } for message in state.get('messages', [])],
        "raw_completion": {
            'type': state['raw_completion'].type,
            "content": state['raw_completion'].content,
            "additional_kwargs": state['raw_completion'].additional_kwargs,
            "response_metadata": state['raw_completion'].response_metadata
        } if state.get('raw_completion') else None,
        "completion": state.get('completion'),
        "compilation_result": state.get('compilation_result'),
        "compilation_attempts": state.get('compilation_attempts'),
        "judgment": state.get('judgment'),
        "judge_output": state.get('judge_output'),
        "judge_attempts": state.get('judge_attempts'),
        "final_state": state.get('final_state')
    }


def visualize_state(state: RAGState) -> None:
    # Check if we are in a Jupyter environment
    try:
        from IPython.display import display, Markdown
        in_jupyter = True
    except ImportError:
        in_jupyter = False

    # Visualization when in a Jupyter environment
    if in_jupyter:
        # Display Challenge Path and Question
        if state.get('challenge_path'):
            display(
                Markdown(f"**Challenge Path:** `{state['challenge_path']}`"))
        if state.get('question'):
            display(Markdown(f"**Question:** `{state['question']}`"))
        if state.get('question_type'):
            display(Markdown(f"**Question Type:** `{state['question_type']}`"))

        # Display Ground Truth and Reference Answer
        if state.get('ground_truth'):
            display(
                Markdown("**Ground Truth Answer:**\n\n" +
                         state['ground_truth']))

        if state.get('ref'):
            display(Markdown(f"**Reference Answer:** `{state['ref']}`"))

        # Display Retrieved Documents
        if state.get('docs'):
            display(Markdown("**Retrieved Documents:**"))
            for doc in state['docs']:
                # Assuming each document has a `text` attribute for its content
                # Show a preview of the document
                if hasattr(doc, 'page_content'):
                    display(Markdown(f"- `{doc.page_content[:200]}...`"))
                else:
                    display(Markdown(f"- `{doc['page_content'][:200]}...`"))

        # Display AI Raw Completion and Final Completion
        if state.get('raw_completion'):
            display(
                Markdown(f"**Raw Completion:** `{state['raw_completion']}`"))
        if state.get('completion'):
            display(Markdown("**Final Completion:**\n\n```envision\n"
                             + state['completion'] + "\n```"))

        # Display Compilation Results (for coding challenges)
        if state.get('compilation_result'):
            compilation_result = state['compilation_result']
            display(
                Markdown(f"**Compilation Result:** `{compilation_result['valid']}`"))
            if compilation_result.get('messages'):
                display(Markdown(f"**Compilation Messages:**\n" +
                        "\n".join(compilation_result['messages'])))

        # Display Judgment
        if state.get('judgment'):
            display(Markdown(f"**Judgment:** `{state['judgment']}`"))
        if state.get('judge_output'):
            display(Markdown(f"**Judge Output:** `{state['judge_output']}`"))

        # Display Messages if they exist
        if state.get('messages'):
            display(Markdown("**Messages:**"))
            for message in state['messages']:
                # Assuming the 'message' has a 'content' attribute to display
                if hasattr(message, 'content'):
                    display(Markdown(f"- `{message.content}`"))
                else:
                    display(Markdown(f"- `{message['content']}`"))

    else:
        # Fallback to console plain text visualization
        print("Challenge Path:", state.get('challenge_path', 'N/A'))
        print("Question:", state.get('question', 'N/A'))
        print("Question Type:", state.get('question_type', 'N/A'))

        print("\nGround Truth Answer:", state.get('ground_truth', 'N/A'))
        print("Reference Answer:", state.get('ref', 'N/A'))

        print("\nRetrieved Documents:")
        docs = state.get('docs', [])
        if docs:
            for doc in docs:
                # Preview first 200 characters
                if hasattr(doc, 'page_content'):
                    print(f"- {doc.page_content[:200]}...")
                else:
                    print(f"- {doc['page_content'][:200]}...")
        else:
            print("No documents retrieved.")

        print("\nRaw Completion:", state.get('raw_completion', 'N/A'))
        print("Final Completion:", state.get('completion', 'N/A'))

        if state.get('compilation_result'):
            compilation_result = state['compilation_result']
            print("\nCompilation Result Valid:",
                  compilation_result.get('valid', 'N/A'))
            if compilation_result.get('messages'):
                print("Compilation Messages:", "\n".join(
                    compilation_result['messages']))

        print("\nJudgment:", state.get('judgment', 'N/A'))
        print("Judge Output:", state.get('judge_output', 'N/A'))

        # Display Messages if they exist
        if state.get('messages'):
            print("\nMessages:")
            for message in state['messages']:
                # Assuming the 'message' has a 'content' attribute to display
                if message.content:
                    print(f"- {message.content}")
                else:
                    print(f"- {message}")
