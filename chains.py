"""
chains.py
Builds the ChatOpenAI model, a reusable chain for the structured
assessment, a raw System/Human/AI message demo, and a streaming
generator for the narrative.

Compatible with current LangChain (0.3+) / langchain-core / langchain-openai.

Note on LLMChain: the assignment asks for an LLMChain specifically, but
LLMChain lives in the base `langchain` package (not `langchain-core`), is
deprecated upstream, and is sometimes missing if only the split packages
(langchain-core / langchain-openai / langchain-community) were installed.
To avoid crashing the app in that situation, this module tries LLMChain
first and automatically falls back to the modern LCEL equivalent
(`prompt | llm`), which does the same job. `run_assessment_chain()` below
hides that difference so the rest of the app never needs to know which
one is active.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.prompts import ASSESSMENT_PROMPT_TEMPLATE, NARRATIVE_CHAT_TEMPLATE, SYSTEM_SAFETY_PROMPT

try:
    from langchain.chains import LLMChain
    _HAS_LLMCHAIN = True
except ImportError:
    LLMChain = None
    _HAS_LLMCHAIN = False


def build_llm(api_key: str, model_name: str, temperature: float = 0.3, streaming: bool = False) -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for this app."""
    return ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        streaming=streaming,
    )


def build_assessment_chain(llm: ChatOpenAI):
    """
    Reusable chain that turns patient inputs into a JSON assessment.
    Returns a real LLMChain if the `langchain` package is installed,
    otherwise an equivalent LCEL runnable (ASSESSMENT_PROMPT_TEMPLATE | llm).
    """
    if _HAS_LLMCHAIN:
        return LLMChain(llm=llm, prompt=ASSESSMENT_PROMPT_TEMPLATE)
    return ASSESSMENT_PROMPT_TEMPLATE | llm


def run_assessment_chain(chain, inputs: dict) -> str:
    """
    Invokes the chain built by build_assessment_chain() and returns the
    raw text response, regardless of whether it's a legacy LLMChain
    (returns a dict with a "text" key) or an LCEL runnable (returns an
    AIMessage with a .content attribute).
    """
    result = chain.invoke(inputs)
    if isinstance(result, dict) and "text" in result:
        return result["text"]
    if hasattr(result, "content"):
        return result.content
    return str(result)


def run_message_demo(llm: ChatOpenAI, patient_summary: str) -> str:
    """
    Demonstrates working directly with SystemMessage / HumanMessage / AIMessage,
    as required by the assignment (section 7). Returns the assistant's reply text.
    """
    messages = [
        SystemMessage(content=SYSTEM_SAFETY_PROMPT),
        HumanMessage(content=f"In one sentence, acknowledge this patient info: {patient_summary}"),
    ]
    response: AIMessage = llm.invoke(messages)
    return response.content


def stream_narrative(llm: ChatOpenAI, inputs: dict):
    """
    Generator that yields narrative text chunks for st.write_stream.
    `llm` should be built with streaming=True.
    """
    messages = NARRATIVE_CHAT_TEMPLATE.format_messages(**inputs)
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
