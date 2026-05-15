from __future__ import annotations

from dotenv import load_dotenv
import os
import string
import sys
from pathlib import Path
from textwrap import shorten

from openai import OpenAI

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def show_section(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def import_llamaindex_components():
    try:
        from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI as LlamaOpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing required LlamaIndex packages. Install them with:\n"
            'pip install pypdf "llama-index-core==0.14.10" '
            "llama-index-embeddings-openai llama-index-llms-openai"
        ) from exc

    return Settings, SimpleDirectoryReader, VectorStoreIndex, OpenAIEmbedding, LlamaOpenAI


def find_groundwork_docs_dir() -> Path | None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    candidate_dirs = [
        script_dir / "resources" / "groundwork_docs",
        repo_root / "assignments_06" / "resources" / "groundwork_docs",
        repo_root / "lessons" / "06_AI_augmentation" / "resources" / "groundwork_docs",
        repo_root.parent / "lessons" / "06_AI_augmentation" / "resources" / "groundwork_docs",
        Path.cwd() / "assignments_06" / "resources" / "groundwork_docs",
        Path.cwd() / "lessons" / "06_AI_augmentation" / "resources" / "groundwork_docs",
        Path.cwd().parent / "lessons" / "06_AI_augmentation" / "resources" / "groundwork_docs",
    ]

    for candidate in candidate_dirs:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_dir():
            return resolved

    search_root = repo_root.parent
    skip_dirs = {".git", ".venv", "node_modules", "__pycache__"}
    for current_root, dirs, _ in os.walk(search_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        if "groundwork_docs" in dirs:
            return Path(current_root) / "groundwork_docs"

    return None


def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "in",
        "on",
        "of",
        "for",
        "to",
        "is",
        "are",
        "was",
        "were",
        "by",
        "with",
        "at",
        "from",
        "that",
        "this",
        "as",
        "be",
        "it",
        "its",
        "their",
        "they",
        "we",
        "you",
        "our",
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator) for w in query.lower().split() if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(
        ((name, content) for score, name, content in scores if score > 0), None
    )
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]

    if verbose:
        print("\nNo overlapping keywords found.")
    return [("None found", "No relevant content.")]


def load_keyword_documents(docs_dir: Path) -> dict[str, str]:
    return {
        file_path.name: file_path.read_text(encoding="utf-8")
        for file_path in sorted(docs_dir.glob("*.txt"))
    }


def node_filename(source_node) -> str:
    return source_node.node.metadata.get("file_name", "Unknown file")


def preview_text(text: str, width: int = 200) -> str:
    normalized = " ".join(text.split())
    return shorten(normalized, width=width, placeholder="...")


def print_top_source_node(source_nodes) -> None:
    if not source_nodes:
        print("Top source node: None")
        return

    top_node = source_nodes[0]
    print(f"Top source document: {node_filename(top_node)}")
    print(f"Top source similarity score: {top_node.score:.4f}")
    print(f"Top source chunk preview: {preview_text(top_node.node.get_content())}")


def print_all_source_nodes(source_nodes) -> None:
    if not source_nodes:
        print("No source nodes were returned.")
        return

    for i, source_node in enumerate(source_nodes, start=1):
        print(f"Source node {i} document: {node_filename(source_node)}")
        print(f"Source node {i} similarity score: {source_node.score:.4f}")
        print(
            f"Source node {i} chunk preview: "
            f"{preview_text(source_node.node.get_content())}"
        )
        print()


def answer_with_keyword_rag(
    client: OpenAI, question: str, keyword_documents: dict[str, str]
) -> dict[str, str]:
    if not keyword_documents:
        return {
            "source_name": "None found",
            "answer": "No .txt documents were available for the keyword comparison.",
            "source_preview": "No relevant content.",
        }

    best_match = simple_keyword_retrieval(question, keyword_documents, verbose=False)[0]
    source_name, source_text = best_match

    if source_name == "None found":
        return {
            "source_name": source_name,
            "answer": "I could not find overlapping keywords for that question in the Groundwork text documents.",
            "source_preview": source_text,
        }

    prompt = f"""
You are answering questions about Groundwork Coffee Co.
Use only the provided context. If the context does not clearly answer the
question, say that the answer is not stated in the document.

Question: {question}

Document name: {source_name}
Context:
{source_text}
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the provided Groundwork Coffee Co. context. "
                    "Do not invent details."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    answer = response.choices[0].message.content.strip()
    return {
        "source_name": source_name,
        "answer": answer,
        "source_preview": preview_text(source_text),
    }


def print_side_by_side_comparison(
    question: str,
    keyword_result: dict[str, str],
    semantic_response,
) -> None:
    print(f"Question: {question}")
    print("Keyword RAG response:")
    print(f"Answer: {keyword_result['answer']}")
    print(f"Retrieved document: {keyword_result['source_name']}")
    print(f"Retrieved text preview: {keyword_result['source_preview']}")
    print()
    print("Semantic RAG response:")
    print(f"Answer: {str(semantic_response).strip()}")
    print_top_source_node(semantic_response.source_nodes)
    print()


def print_new_document_example() -> None:
    print("Suggested Extension C document: seasonal_specials_update.txt")
    print("Suggested test query: What seasonal drinks are available this month?")
    print()


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Add it to your .env file.")

    client = OpenAI()
    (
        Settings,
        SimpleDirectoryReader,
        VectorStoreIndex,
        OpenAIEmbedding,
        LlamaOpenAI,
    ) = import_llamaindex_components()

    docs_dir = find_groundwork_docs_dir()
    assert docs_dir is not None and docs_dir.exists(), (
        "Document directory not found. Checked common lesson/resource paths for "
        "groundwork_docs near this repository."
    )

    # Step 2: Load the Documents
    show_section("LOAD DOCUMENTS")
    print(f"Groundwork docs directory: {docs_dir}")
    documents = SimpleDirectoryReader(str(docs_dir)).load_data()
    print(f"Loaded {len(documents)} document(s).")
    for document in documents:
        print(f"- {document.metadata.get('file_name', 'Unknown file')}")
    print()

    # Step 3: Build the Index and Query Engine
    show_section("BUILD INDEX")
    Settings.llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=3)
    print("Index built successfully. Ready to answer questions.")
    print()

    # Step 4: Query the Assistant
    show_section("GROUNDWORK Q&A")
    questions = [
        "What are Groundwork's hours on weekends?",
        "Do you offer any dairy-free milk options?",
        "How does the loyalty program work?",
        "How did Groundwork Coffee get started?",
        "Do you offer catering or wholesale orders?",
    ]

    for question in questions:
        response = query_engine.query(question)
        print(f"Question: {question}")
        print(f"Answer: {str(response).strip()}")
        print_top_source_node(response.source_nodes)
        print()

    # Reflection after the five standard questions:
    # The assistant generally sounded confident and accurate on the five standard
    # questions, and the retrievals lined up well with the intended documents.
    # The clearest strength showed up on the catering-or-wholesale question:
    # semantic retrieval produced a combined answer that covered both services,
    # which is harder for a single exact-match document lookup to do well.

    # Step 5: Find a Failure
    show_section("FAILURE CASE")
    hard_question = (
        "If I place a catering order on Sunday morning, can I earn loyalty points "
        "and pick it up before the cafe opens?"
    )
    failure_response = query_engine.query(hard_question)
    print(f"Question: {hard_question}")
    print(f"Answer: {str(failure_response).strip()}")
    print_all_source_nodes(failure_response.source_nodes)

    # Failure reflection:
    # I chose this question because it likely requires combining multiple
    # documents: weekend hours, catering details, and loyalty rules. It may also
    # require the model to avoid making assumptions if the documents do not
    # explicitly connect those policies.
    # What went wrong: the model inferred that catering orders would not earn
    # loyalty points because the loyalty document said points apply to qualifying
    # cafe purchases, but the documents never explicitly addressed catering
    # rewards. It also focused on opening hours instead of emphasizing the
    # 48-hour catering lead-time rule, so the answer was only partially grounded.
    #
    # Tone observation: the model still sounded confident even though part of the
    # answer was not directly grounded in the documents. That is a useful reminder
    # that confident wording is not proof that an answer is fully supported.
    #
    # Improvement idea: I would tighten the prompt so the assistant must clearly
    # say when a document set does not explicitly answer one part of a question. I
    # would also consider structured citation checks or an evaluator pass for
    # compound questions that span several documents.

    # Extension A: Side-by-Side Comparison
    show_section("EXTENSION A - KEYWORD VS SEMANTIC RAG")
    keyword_documents = load_keyword_documents(docs_dir)
    print(f"Loaded {len(keyword_documents)} text document(s) for keyword retrieval.")
    print()

    for question in questions:
        keyword_result = answer_with_keyword_rag(client, question, keyword_documents)
        semantic_response = query_engine.query(question)
        print_side_by_side_comparison(question, keyword_result, semantic_response)

    # Extension A comparison notes:
    # Keyword RAG clearly failed on the weekend-hours query because it retrieved
    # wholesale.txt instead of hours.txt, showing how brittle exact token overlap
    # can be. It also underperformed on the catering-or-wholesale query because it
    # focused on wholesale.txt and missed the catering half of the question.
    #
    # Keyword RAG did just as well as semantic RAG on straightforward queries
    # where the exact language appeared in a single document, especially the
    # loyalty-program and company-story questions. It also did well on the
    # dairy-free milk question once it matched the menu text directly.
    #
    # Semantic RAG was stronger overall because it handled combined intent better
    # and was less dependent on the exact phrasing used in the query. Even when it
    # retrieved an unexpected top chunk, it still produced a grounded answer more
    # reliably than the keyword baseline.

    # Extension C: Add a New Document
    # This project rebuilds the index from all files in groundwork_docs every time
    # the script runs, so any new document you add to that folder is automatically
    # included in retrieval the next time you run the script.
    show_section("EXTENSION C - NEW DOCUMENT TEST")
    seasonal_question = "What seasonal drinks are available this month?"
    seasonal_response = query_engine.query(seasonal_question)
    print(f"Question: {seasonal_question}")
    print(f"Answer: {str(seasonal_response).strip()}")
    print_top_source_node(seasonal_response.source_nodes)
    print()

    # Extension C reflection:
    # I added seasonal_specials_update.txt, which contains a short cafe update
    # listing three seasonal drinks, one seasonal pastry, and a note that the
    # specials are available while supplies last.
    #
    # I tested it with the query "What seasonal drinks are available this month?"
    # and the assistant retrieved the new seasonal document as the top source
    # node. That demonstrates a key advantage of RAG over fine-tuning: new
    # business information can be added by updating the document set and
    # rebuilding the index, without retraining the model on a new dataset.


if __name__ == "__main__":
    main()


# --- Reflection ---
#
# The equivalent LlamaIndex setup in this project takes about 5 core lines:
# loading documents, setting the LLM, setting the embedding model, building the
# index, and creating the query engine. Even with printing and helper functions
# around it, the actual framework-driven RAG setup is dramatically shorter than a
# fully manual semantic RAG pipeline. That shows the value of a framework: it
# removes a lot of plumbing so you can spend more time inspecting documents,
# testing retrieval quality, and evaluating failure cases.
#
# A different high-value use case would be an HR or benefits assistant for a
# company with employee handbooks, PTO policies, insurance summaries, and leave
# documents. Instead of manually searching PDFs, employees could ask grounded
# questions about enrollment deadlines, coverage rules, or holiday policy.
#
# One failure mode RAG cannot fully prevent is a misleading answer built from
# retrieved text that is incomplete, ambiguous, or internally inconsistent. Even
# when retrieval works correctly, the model can still overgeneralize, merge ideas
# too aggressively, or answer with more certainty than the source material really
# supports.
