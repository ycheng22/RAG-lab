"""Prompts and LCEL chains (notebook 04)."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from raglab import llm as llm_module

NOT_FOUND = "I couldn't find that in the Token Hotels documents."


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks as numbered blocks the model can cite: "[1] (source_file) text"."""
    return "\n\n".join(
        f"[{i}] ({d.metadata.get('source_file', 'unknown')}) {d.page_content}" for i, d in enumerate(docs, start=1)
    )


BASIC_PROMPT = ChatPromptTemplate.from_template(
    """Use the context to answer the question.

Context:
{context}

Question: {question}"""
)

GROUNDED_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the guest concierge for Token Hotels & Resorts. Answer using ONLY the numbered "
            "context passages. Never use outside knowledge, and never guess. If the passages do not "
            f'contain the answer, reply with exactly: "{NOT_FOUND}" '
            "Keep answers short and factual. If two passages disagree, prefer the most recent one "
            "and say so.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def build_rag_chain(retriever: BaseRetriever, prompt: ChatPromptTemplate = GROUNDED_PROMPT, llm=None) -> Runnable:
    """question (str) -> answer (str):
    {"context": retriever | format_docs, "question": passthrough} | prompt | llm | StrOutputParser()"""
    llm = llm or llm_module.get_lc_chat()
    return (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def build_rag_chain_with_sources(
    retriever: BaseRetriever, prompt: ChatPromptTemplate = GROUNDED_PROMPT, llm=None
) -> Runnable:
    """question (str) -> {"answer": str, "contexts": list[Document]}, retrieving only once.
    Returning the contexts is what makes evaluation (notebook 05) and citations (notebook 10) possible."""
    llm = llm or llm_module.get_lc_chat()
    generate = (
        RunnableLambda(lambda x: {"context": format_docs(x["contexts"]), "question": x["question"]})
        | prompt
        | llm
        | StrOutputParser()
    )
    return (
        RunnableParallel(contexts=retriever, question=RunnablePassthrough())
        .assign(answer=generate)
        .pick(["answer", "contexts"])
    )
