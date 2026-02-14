#!/usr/bin/env python3
"""
LangGraph Agentic RAG — Indian Constitution Q&A

Agentic RAG pipeline using LangGraph for orchestration. Performs iterative
retrieval with sufficiency checking, query reformulation, and decomposition.
"""

# ── Imports ───────────────────────────────────────────────
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import TypedDict, Literal

import chromadb
import fitz  # pymupdf
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from openai import OpenAI

load_dotenv(dotenv_path="../.env")

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
LLM_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

# ── Prompt Templates ──────────────────────────────────────

SUFFICIENCY_PROMPT = (
    "You are evaluating whether retrieved context is sufficient to FULLY answer a question "
    "about the Indian Constitution.\n\n"
    "Question: {question}\n\n"
    "Retrieved context:\n{context}\n\n"
    "TASK: Determine if the context contains ALL the information needed for a COMPLETE answer.\n\n"
    "STEP 1: Break down the question — what distinct pieces of information are needed?\n"
    "  - If the question asks about a single fact, identify that fact.\n"
    "  - If the question compares, contrasts, or asks about multiple entities/concepts,\n"
    "    list each entity/concept that needs coverage.\n"
    "  - If the question has multiple parts, list each part.\n\n"
    "STEP 2: For EACH piece identified, check whether the context contains it.\n\n"
    "STEP 3: Decision:\n"
    "  - SUFFICIENT only if ALL parts are covered. The context does not need to use the\n"
    "    same wording — legal text often states facts indirectly (e.g. a prerequisite\n"
    "    implies a requirement). That counts.\n"
    "  - INSUFFICIENT if ANY part is missing or only partially covered.\n\n"
    "Respond with ONLY a JSON object:\n"
    '{{"parts_needed": ["<part 1>", "<part 2>", ...], '
    '"parts_found": ["<part 1>", ...], '
    '"parts_missing": ["<part N>", ...], '
    '"sufficient": true/false, '
    '"reason": "<brief explanation>"}}'
)

REFORMULATE_PROMPT = (
    "You are a search query optimizer for the Indian Constitution.\n\n"
    "Original question: {question}\n"
    "{prev_text}\n\n"
    "Generate a single improved search query that will retrieve more relevant "
    "constitutional text. Use different keywords, synonyms, or focus on a "
    "specific aspect of the question.\n\n"
    "Return ONLY the query string, nothing else."
)

DECOMPOSE_PROMPT = (
    "You are a query decomposition engine for the Indian Constitution.\n\n"
    "Question: {question}\n\n"
    "{previous_context}"
    "DECOMPOSITION STRATEGY - TWO STAGES:\n\n"
    "STAGE 1: If this is a COMPARISON question (comparing entities, asking 'same way', 'difference'), "
    "FIRST split it into separate questions for EACH entity.\n\n"
    "STAGE 2: Then, for EACH entity-specific question, break it down into GRANULAR sub-questions "
    "focusing on KEY CONCEPTS and KEYWORDS (e.g., entity name, specific powers, article numbers, limitations).\n\n"
    "EXAMPLE:\n"
    'Question: "How does Entity A\'s power differ from Entity B\'s power?"\n'
    'Stage 1: Split → "What is Entity A\'s power?" + "What is Entity B\'s power?"\n'
    'Stage 2: Decompose each into granular questions about keywords (entity names, specific powers, relevant article numbers, limitations, etc.)\n\n'
    "Return ALL granular sub-questions from Stage 2 as a JSON array.\n\n"
    "REQUIREMENTS:\n"
    "- For comparisons: FIRST split entities, THEN decompose each\n"
    "- Each sub-question focuses on ONE key concept/keyword\n"
    "- Use specific article numbers when relevant\n"
    "- Generate 4-8 granular sub-questions\n"
    "- Return ONLY a JSON array of strings\n"
)

DECISION_PROMPT = (
    "You are a decision engine for a RAG (Retrieval-Augmented Generation) system.\n\n"
    "Question: {question}\n\n"
    "Context Status: {sufficiency}\n"
    "Iteration: {iteration_count}/{max_iterations}\n"
    "Documents Retrieved: {num_docs}\n"
    "Previously Decomposed Queries: {num_decomposed}\n\n"
    "Based on the context status and current state, decide the next step:\n"
    "- 'answer': If we should generate the final answer (even if context is insufficient)\n"
    "- 'refine_simple': If we should reformulate the query and search again\n"
    "- 'refine_complex': If we should decompose the question into sub-questions\n\n"
    "Consider:\n"
    "- If context is insufficient, should we refine or decompose?\n"
    "- If we've tried many decompositions, should we try reformulating instead?\n"
    "- If we've tried many reformulations, should we try decomposing instead?\n"
    "- The complexity of the question\n\n"
    "Return ONLY one word: 'answer', 'refine_simple', or 'refine_complex'"
)

ANSWER_PROMPT = (
    "You are an expert on the Indian Constitution. Answer the question using the "
    "retrieved context provided below.\n\n"
    "Question: {question}\n\n"
    "Retrieved Context:\n{context}\n\n"
    "INSTRUCTIONS:\n"
    "- Ground every claim in the context. ONLY state what the context explicitly says.\n"
    "- Use the exact terms and wording from the constitutional text when stating key facts\n"
    "  (e.g. if it says 'thirty-five years', say 'thirty-five years').\n"
    "- Provide a clear, direct answer FIRST, then cite the correct article numbers.\n"
    "- Double-check that article numbers you cite actually correspond to the topic\n"
    "  (e.g. do not cite the article that defines the office when asked about the oath).\n"
    "- Only say information is missing if it is truly absent from the context.\n\n"
    "Use markdown to format your answer."
)

# ── Logging Helper ────────────────────────────────────────


def setup_logging(log_filename: str):
    """Configure logging with file + console handlers. Suppresses noisy HTTP loggers."""
    for noisy in ("httpcore", "httpx", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)


# ── Initialization ────────────────────────────────────────

openai_client = OpenAI()
together_client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)

# Load and chunk the constitution PDF
doc = fitz.open("indiaconstitution.pdf")
pages = [page.get_text() for page in doc]
all_text = "\n".join(pages)
print(f"Total pages: {len(doc)}")
print(f"Total characters: {sum(len(p) for p in pages):,}")


def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    resp = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in resp.data]


chunks = chunk_text(all_text)
print(f"{len(chunks)} chunks (avg {sum(len(c) for c in chunks) // len(chunks)} chars)")

# Build ChromaDB vector index
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="langgraph_constitution",
    metadata={"hnsw:space": "cosine"},
)

batch_size = 500
for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    ids = [f"chunk_{j}" for j in range(i, i + len(batch))]
    embeddings = embed(batch)
    collection.upsert(ids=ids, documents=batch, embeddings=embeddings)
    print(f"  Embedded batch {i // batch_size + 1} ({len(batch)} chunks)")

print(f"Total in collection: {collection.count()}")

# ── State Management ──────────────────────────────────────

seen_hashes = set()
prev_queries = []


def reset_state():
    """Reset global state for a new question."""
    global seen_hashes, prev_queries
    seen_hashes = set()
    prev_queries = []


def _add_docs(docs: list[str], accumulated_docs: list[str]) -> list[str]:
    """Add new docs to accumulated list, avoiding duplicates."""
    for doc in docs:
        h = hash(doc)
        if h not in seen_hashes:
            seen_hashes.add(h)
            accumulated_docs.append(doc)
    return accumulated_docs


# ── Tool Functions ────────────────────────────────────────


def semantic_search(query: str, n_results: int = 5) -> str:
    """Search the Indian Constitution using semantic similarity."""
    q_emb = embed([query])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=n_results)
    docs = results["documents"][0]
    prev_queries.append(query)
    return "\n\n---\n\n".join(docs)


def check_sufficiency(question: str, accumulated_docs: list[str]) -> str:
    """Check if accumulated context is sufficient to answer the question."""
    logger.debug(f"[check_sufficiency] Evaluating with {len(accumulated_docs)} docs")

    context = "\n\n---\n\n".join(accumulated_docs)
    if not context:
        return "INSUFFICIENT: No context retrieved yet."

    prompt = SUFFICIENCY_PROMPT.format(question=question, context=context)

    resp = together_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            if result.get("sufficient", True):
                logger.debug("[check_sufficiency] Result: SUFFICIENT")
                return "SUFFICIENT"
            reason = result.get("reason", "Context does not fully address the question.")
            logger.debug(f"[check_sufficiency] Result: INSUFFICIENT - {reason}")
            return f"INSUFFICIENT: {reason}"
    except (json.JSONDecodeError, ValueError) as e:
        logger.debug(f"[check_sufficiency] Failed to parse JSON: {e}")

    return "SUFFICIENT"  # default fallback


def reformulate_query(question: str) -> str:
    """Generate a better search query when initial retrieval didn't find enough."""
    logger.debug(f"[reformulate_query] Reformulating: {question[:100]}...")

    prev_text = ""
    if prev_queries:
        prev_text = (
            "\n\nPrevious queries already tried (generate something DIFFERENT):\n"
            + "\n".join(f"- {q}" for q in prev_queries)
        )

    prompt = REFORMULATE_PROMPT.format(question=question, prev_text=prev_text)

    resp = together_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    new_query = resp.choices[0].message.content.strip().strip("\"'")

    if new_query.lower() in [q.lower() for q in prev_queries]:
        return (
            f"DUPLICATE: The reformulated query '{new_query}' was already tried. "
            "Consider using decompose_query instead."
        )

    logger.debug(f"[reformulate_query] New query: {new_query}")
    return new_query


def decompose_query(question: str, previous_decompositions: list[str] = None) -> str:
    """Break a complex question into simpler sub-questions."""
    logger.debug(f"[decompose_query] Decomposing: {question[:100]}...")

    previous_context = ""
    if previous_decompositions:
        logger.debug(f"[decompose_query] {len(previous_decompositions)} sub-questions already tried")
        previous_context = (
            "\n\nIMPORTANT: The following sub-questions have ALREADY been searched. "
            "Generate DIFFERENT sub-questions with different wording or angles:\n"
            + "\n".join(f"- {q}" for q in previous_decompositions[:5])
            + "\n\nGenerate NEW sub-questions that are NOT similar to the above.\n"
        )

    prompt = DECOMPOSE_PROMPT.format(question=question, previous_context=previous_context)

    resp = together_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    logger.debug(f"[decompose_query] Raw LLM response: {raw[:200]}...")

    try:
        match = re.search(r"\[(?:[^\[\]]+|\[[^\]]*\])*\]", raw, re.DOTALL)
        if not match:
            match = re.search(r"\[.*?\]", raw, re.DOTALL)

        if match:
            json_str = match.group().strip()
            json_str = re.sub(r"^```(?:json)?\s*", "", json_str)
            json_str = re.sub(r"\s*```$", "", json_str)
            sub_qs = json.loads(json_str)
            if isinstance(sub_qs, list) and len(sub_qs) >= 1 and all(isinstance(q, str) for q in sub_qs):
                result = json.dumps(sub_qs[:6])
                logger.debug(f"[decompose_query] Decomposed into {len(sub_qs[:6])} sub-questions")
                for i, q in enumerate(sub_qs[:6], 1):
                    logger.debug(f"  {i}. {q[:100]}...")
                return result
        else:
            logger.debug("[decompose_query] Invalid format: not a list of strings")
    except (json.JSONDecodeError, ValueError) as e:
        logger.debug(f"[decompose_query] Failed to parse JSON: {e}")
        logger.debug(f"[decompose_query] Raw response was: {raw[:300]}")

    logger.debug("[decompose_query] Falling back to original question")
    return json.dumps([question])


# ── LangGraph State & Nodes ───────────────────────────────


class AgentState(TypedDict):
    question: str
    accumulated_docs: list[str]
    search_results: str
    sufficiency_result: str
    reformulated_query: str
    decomposed_queries: list[str]
    answer: str
    iteration_count: int
    max_iterations: int


def initial_search_node(state: AgentState) -> AgentState:
    """Execute semantic_search with the original question."""
    logger.info(f"[NODE: initial_search] Searching: {state['question']}...")

    results = semantic_search(state["question"], n_results=5)
    accumulated_docs = _add_docs(results.split("\n\n---\n\n"), state.get("accumulated_docs", []))

    return {
        "search_results": results,
        "accumulated_docs": accumulated_docs,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def check_sufficiency_node(state: AgentState) -> AgentState:
    """Check if accumulated context is sufficient."""
    logger.info(f"[NODE: check_sufficiency] Evaluating with {len(state.get('accumulated_docs', []))} docs")
    result = check_sufficiency(state["question"], state.get("accumulated_docs", []))
    return {"sufficiency_result": result}


def decide_next_step_node(state: AgentState) -> Literal["answer", "refine_simple", "refine_complex"]:
    """Decide next step based on sufficiency result."""
    sufficiency = state.get("sufficiency_result", "")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 5)

    if sufficiency.startswith("SUFFICIENT"):
        logger.info("[DECISION] → answer (context is sufficient)")
        return "answer"

    if iteration_count >= max_iterations:
        logger.info(f"[DECISION] → answer (max iterations reached: {iteration_count}/{max_iterations})")
        return "answer"

    prompt = DECISION_PROMPT.format(
        question=state["question"],
        sufficiency=sufficiency,
        iteration_count=iteration_count,
        max_iterations=max_iterations,
        num_docs=len(state.get("accumulated_docs", [])),
        num_decomposed=len(state.get("decomposed_queries", [])),
    )

    resp = together_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    decision = resp.choices[0].message.content.strip().lower()

    if "answer" in decision:
        decision = "answer"
    elif "refine_complex" in decision or "decompose" in decision or "complex" in decision:
        decision = "refine_complex"
    elif "refine_simple" in decision or "reformulate" in decision or "simple" in decision:
        decision = "refine_simple"
    else:
        decision = "refine_simple"

    logger.info(f"[DECISION] → {decision}")
    return decision


def refine_simple_node(state: AgentState) -> AgentState:
    """Reformulate query and search again."""
    logger.info("[NODE: refine_simple] Reformulating and re-searching")

    new_query = reformulate_query(state["question"])

    if new_query.startswith("DUPLICATE"):
        logger.debug("[refine_simple] Got duplicate, switching to decompose")
        return {"reformulated_query": new_query}

    results = semantic_search(new_query, n_results=5)
    accumulated_docs = _add_docs(results.split("\n\n---\n\n"), state.get("accumulated_docs", []))

    return {
        "reformulated_query": new_query,
        "search_results": results,
        "accumulated_docs": accumulated_docs,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def refine_complex_node(state: AgentState) -> AgentState:
    """Decompose query and search for each sub-question."""
    logger.info("[NODE: refine_complex] Decomposing and searching sub-questions")

    previously_searched = state.get("decomposed_queries", [])
    previously_searched_lower = {q.lower().strip() for q in previously_searched}
    logger.debug(f"[refine_complex] Previously searched {len(previously_searched)} sub-questions")

    sub_questions_json = decompose_query(state["question"], previous_decompositions=previously_searched)

    try:
        sub_questions = json.loads(sub_questions_json)
        if not isinstance(sub_questions, list) or len(sub_questions) == 0:
            raise ValueError("Invalid sub-questions format")
    except (json.JSONDecodeError, ValueError) as e:
        logger.debug(f"[refine_complex] Failed to parse decomposed queries: {e}")
        sub_questions = [state["question"]]

    # Deduplicate against previously searched
    seen_in_batch = set()
    new_sub_questions = []
    for q in sub_questions:
        q_lower = q.lower().strip()
        if len(q_lower) > 10 and q_lower not in seen_in_batch and q_lower not in previously_searched_lower:
            seen_in_batch.add(q_lower)
            new_sub_questions.append(q)
        elif q_lower in previously_searched_lower:
            logger.debug(f'[refine_complex] Skipping duplicate: "{q[:60]}..."')

    if not new_sub_questions:
        logger.debug("[refine_complex] All sub-questions were duplicates, using original as fallback")
        new_sub_questions = [state["question"]]

    all_searched_queries = previously_searched + new_sub_questions
    logger.info(f"[refine_complex] Searching {len(new_sub_questions[:6])} new sub-questions (total: {len(all_searched_queries)})")

    all_results = []
    accumulated_docs = state.get("accumulated_docs", [])
    docs_before = len(accumulated_docs)

    for i, sub_q in enumerate(new_sub_questions[:6], 1):
        logger.info(f'[refine_complex] [{i}/{len(new_sub_questions[:6])}] Searching: "{sub_q[:80]}..."')
        results = semantic_search(sub_q, n_results=5)
        new_docs = results.split("\n\n---\n\n")
        accumulated_docs = _add_docs(new_docs, accumulated_docs)
        all_results.append(f"Sub-question {i}: {sub_q}\n\n{results}")
        logger.debug(f"  → Found {len(new_docs)} document chunks")

    new_docs_count = len(accumulated_docs) - docs_before
    logger.info(f"[refine_complex] Done. Docs: {docs_before} → {len(accumulated_docs)} (+{new_docs_count} new)")

    if new_docs_count == 0:
        logger.debug("[refine_complex] WARNING: No new documents found!")

    return {
        "decomposed_queries": all_searched_queries[:15],
        "search_results": "\n\n---\n\n".join(all_results),
        "accumulated_docs": accumulated_docs,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def answer_node(state: AgentState) -> AgentState:
    """Generate final answer using accumulated context."""
    logger.info("[NODE: answer] Generating final answer")

    sufficiency = state.get("sufficiency_result", "")
    is_insufficient = sufficiency.startswith("INSUFFICIENT")

    if is_insufficient and state.get("iteration_count", 0) >= state.get("max_iterations", 3):
        logger.info("[answer] WARNING: Answering with INSUFFICIENT context (max iterations reached)")

    context = "\n\n---\n\n".join(state.get("accumulated_docs", []))
    prompt = ANSWER_PROMPT.format(question=state["question"], context=context)

    resp = together_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    answer = resp.choices[0].message.content.strip()
    logger.info(f"[answer] Generated answer ({len(answer)} chars)")
    if is_insufficient:
        logger.info(f"[answer] Answer generated despite INSUFFICIENT context")

    return {"answer": answer}


# ── Build LangGraph ───────────────────────────────────────

workflow = StateGraph(AgentState)

workflow.add_node("initial_search", initial_search_node)
workflow.add_node("check_sufficiency", check_sufficiency_node)
workflow.add_node("refine_simple", refine_simple_node)
workflow.add_node("refine_complex", refine_complex_node)
workflow.add_node("answer", answer_node)

workflow.set_entry_point("initial_search")
workflow.add_edge("initial_search", "check_sufficiency")
workflow.add_conditional_edges(
    "check_sufficiency",
    decide_next_step_node,
    {
        "answer": "answer",
        "refine_simple": "refine_simple",
        "refine_complex": "refine_complex",
    },
)
workflow.add_edge("refine_simple", "check_sufficiency")
workflow.add_edge("refine_complex", "check_sufficiency")
workflow.add_edge("answer", END)

app = workflow.compile()
print("LangGraph agent created")


# ── Main ──────────────────────────────────────────────────


def main():
    os.makedirs("run-logs", exist_ok=True)
    model_short = LLM_MODEL.split("/")[-1]
    log_filename = f"run-logs/langgraph_agent_{model_short}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    setup_logging(log_filename)

    print(f"Starting LangGraph agent run - log: {log_filename}")
    logger.info(f"Using model: {LLM_MODEL}")
    logger.info("Retrieval: semantic search")
    print("=" * 80)

    reset_state()

    question = "What is the difference in the composition of the electoral college for the President versus the Vice-President?"

    initial_state: AgentState = {
        "question": question,
        "accumulated_docs": [],
        "search_results": "",
        "sufficiency_result": "",
        "reformulated_query": "",
        "decomposed_queries": [],
        "answer": "",
        "iteration_count": 0,
        "max_iterations": 5,
    }

    print(f"\nQuestion: {question}\n")
    print("=" * 80)

    try:
        final_state = initial_state.copy()
        logger.info("Starting graph execution...\n")

        for step in app.stream(initial_state):
            node_name = list(step.keys())[0]
            node_output = step[node_name]
            final_state.update(node_output)

            if "sufficiency_result" in node_output:
                logger.info(f"  Sufficiency: {node_output['sufficiency_result'][:100]}")
            if "answer" in node_output:
                logger.info(f"  Answer length: {len(node_output.get('answer', ''))} chars")

        if final_state:
            print("\n" + "=" * 80)
            print("FINAL ANSWER")
            print("=" * 80 + "\n")
            print(final_state.get("answer", "No answer generated"))

            with open(log_filename, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write("EXECUTION SUMMARY:\n")
                f.write("=" * 80 + "\n")
                f.write(f"Question: {question}\n")
                f.write(f"Iterations: {final_state.get('iteration_count', 0)}\n")
                f.write(f"Documents accumulated: {len(final_state.get('accumulated_docs', []))}\n")
                f.write(f"Sufficiency: {final_state.get('sufficiency_result', 'N/A')}\n")
                f.write(f"\nAnswer:\n{final_state.get('answer', 'N/A')}\n")

            print(f"\nFull execution log saved to: {log_filename}")

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
