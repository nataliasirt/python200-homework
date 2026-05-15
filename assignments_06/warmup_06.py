from __future__ import annotations

from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

import string
import sys
from pathlib import Path
from textwrap import shorten

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def show_section(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def safe_score(value) -> str:
    if value is None:
        return "None"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def print_source_nodes(source_nodes, limit: int = 3) -> None:
    for i, source_node in enumerate(source_nodes[:limit], start=1):
        chunk_text = " ".join(source_node.node.get_content().split())
        preview = shorten(chunk_text, width=150, placeholder="...")
        print(
            f"Source node {i}: score={safe_score(source_node.score)} | text={preview}"
        )


# --- RAG Concepts ---

# Concepts Q1
#
# Scenario A: RAG is the best fit because the legal team's source of truth is a
# large internal PDF library that changes every quarter. Retrieval lets the
# assistant pull current policy text at answer time instead of trying to bake
# changing documents into model weights.
#
# Scenario B: Fine-tuning is the best fit because the main goal is a consistent,
# unusual writing style rather than retrieving facts from changing documents.
# The team already has 3,000 in-house examples, which is the kind of dataset
# that can teach the model a brand voice it would not reliably pick up from a
# prompt alone.
#
# Scenario C: Prompt engineering is the best fit because the task only concerns
# a single short report and does not need a reusable retrieval pipeline. The
# analyst can paste the report or summarize it in the prompt and get the answer
# without the extra complexity of indexing or fine-tuning.

# Concepts Q2
#
# A confidently wrong answer is more harmful because people are more likely to
# act on it instead of double-checking it. "I'm not sure" creates friction and
# encourages verification, while a polished, certain tone can make bad
# information feel trustworthy.
#
# Example: If a medical chatbot confidently tells a patient to combine two
# medications that should not be taken together, the calm and authoritative tone
# makes the bad advice more dangerous than a hesitant answer that clearly signals
# uncertainty.

# Concepts Q3
#
# Correct RAG pipeline order:
# 1. Receive the user's query - The system gets the question it needs to answer.
# 2. Extract text from source documents - Raw document contents are pulled out so
#    they can be indexed.
# 3. Split text into chunks - Large documents are broken into smaller, searchable
#    pieces.
# 4. Convert text chunks into embeddings - Each chunk is turned into a vector
#    that captures its meaning.
# 5. Embed the user's query - The question is converted into the same vector
#    space as the chunks.
# 6. Retrieve the most relevant chunks - The system finds the chunks whose
#    embeddings are closest to the query embedding.
# 7. Inject retrieved chunks into the prompt - The selected evidence is added to
#    the model's context window.
# 8. Generate a response from the LLM - The model answers using the retrieved
#    context as grounding.


def run_concepts_section() -> None:
    show_section("RAG CONCEPTS")
    print("Concepts Q1 answer is included in comments above the section.")
    print("Concepts Q2 answer is included in comments above the section.")
    print("Concepts Q3 ordered pipeline:")
    ordered_steps = [
        "1. Receive the user's query",
        "2. Extract text from source documents",
        "3. Split text into chunks",
        "4. Convert text chunks into embeddings",
        "5. Embed the user's query",
        "6. Retrieve the most relevant chunks",
        "7. Inject retrieved chunks into the prompt",
        "8. Generate a response from the LLM",
    ]
    for step in ordered_steps:
        print(step)
    print()


# --- Keyword RAG ---

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
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# Keyword Q1
#
# The function selects loyalty.txt, not hours.txt. That happens because this
# simple retriever only counts exact token overlap, and "weekend" does not match
# "weekends" while "your" does match both hiring.txt and loyalty.txt. Those two
# documents tie with a score of 1, and loyalty.txt wins after the reverse sort.

# Keyword Q2
#
# The function selects "None found" because none of the query words exactly
# overlap with the menu text. Keyword RAG does not really get this right because
# a human would probably look at the menu, but the retriever cannot connect
# "without caffeine" to coffee-menu items or to the idea of decaf or
# non-coffee drinks. Semantic retrieval would do better because it compares
# meaning instead of exact word overlap.

# Keyword Q3
#
# Prediction before running: I would expect loyalty.txt to be the best answer
# because "sign up" and "rewards" are conceptually close to a loyalty program.
# Actual result: this retriever still returns "None found" because it cannot
# connect rewards with loyalty or sign up with join unless the exact words match.


def run_keyword_section() -> None:
    show_section("KEYWORD RAG")

    documents = {
        "menu.txt": (
            "We serve espresso, lattes, cappuccinos, and cold brew. Pastries "
            "include croissants and muffins baked fresh daily. Oat milk and "
            "almond milk are available."
        ),
        "hours.txt": (
            "We are open Monday through Friday from 7am to 7pm. On weekends "
            "we open at 8am and close at 5pm. We are closed on Thanksgiving "
            "and Christmas Day."
        ),
        "hiring.txt": (
            "We are currently hiring baristas and shift supervisors. Send your "
            "resume to jobs@groundworkcoffee.com."
        ),
        "loyalty.txt": (
            "Join our loyalty program to earn one point per dollar spent. "
            "Redeem 100 points for a free drink of your choice."
        ),
    }

    query_1 = "What are your hours on the weekend?"
    print("Keyword Q1")
    result_1 = simple_keyword_retrieval(query_1, documents, verbose=True)
    print(f"Selected document: {result_1[0][0]}")
    print()

    query_2 = "Do you have anything without caffeine?"
    print("Keyword Q2")
    result_2 = simple_keyword_retrieval(query_2, documents, verbose=True)
    print(f"Selected document: {result_2[0][0]}")
    print()

    query_3 = "How do I sign up for rewards?"
    print("Keyword Q3")
    print("Prediction before running: loyalty.txt")
    result_3 = simple_keyword_retrieval(query_3, documents, verbose=True)
    print(f"Selected document: {result_3[0][0]}")
    print()


# --- Semantic RAG Concepts ---

# Semantic Q1
#
# A vector embedding is a numeric representation of a piece of text that places
# meaning into a coordinate space. Texts with similar ideas end up near each
# other even if they are written with different words.
#
# The chunk with cosine similarity 0.85 is more relevant than the one with 0.30.
# A score closer to 1 means the query and the chunk point in a very similar
# semantic direction, so they are more closely related in meaning.
#
# Semantic search can find relevant text without exact word overlap because the
# embedding captures concepts and relationships, not just the raw string. That
# lets the system connect phrases like "employee perks" and "benefits package"
# even when the wording changes.

# Semantic Q2
#
# | Feature                    | Keyword RAG                       | Semantic RAG                          |
# |----------------------------|-----------------------------------|---------------------------------------|
# | What is compared?          | Exact word overlap                | Embedding vectors / semantic meaning  |
# | What is retrieved?         | Full document                     | Most relevant chunk(s)                |
# | Can it handle synonyms?    | No                                | Yes, often much better                |
# | Storage format             | Plain text dictionary             | Vector index / embedding store        |
# | Relevance score            | Number of overlapping keywords    | Similarity score such as cosine sim   |


def run_semantic_concepts_section() -> None:
    show_section("SEMANTIC RAG CONCEPTS")
    print("Semantic Q1 and Q2 answers are included in comments above the section.")
    print(
        "Key idea: semantic retrieval compares embeddings, so it can match meaning"
    )
    print("even when the exact query words are not present in the retrieved text.")
    print()


# --- LlamaIndex ---

# LlamaIndex Q1 Observation Template
#
# Query 1 retrieved chunks looked relevant overall. The employee benefits guide
# was the top source by a healthy margin, and the company overview plus remote
# work policy appeared as weaker supporting context. The model's answer sounded
# confident and specific because it listed concrete benefits rather than hedging.
#
# Query 2 also retrieved relevant material. The dedicated security policy file
# was the top source, while the company overview and remote work policy showed up
# as related but broader supporting context. Nothing especially unexpected was
# retrieved, although the high-level company overview appeared in both queries
# because it shares general company language with many questions.

# LlamaIndex Q2 Observation Template
#
# The response barely changed between similarity_top_k=1 and similarity_top_k=5
# because the top benefits document already contained the full answer. In this
# run, more context was not harmful, but it also did not materially improve the
# answer. That illustrates the usual tradeoff: additional chunks can help when
# the answer is distributed across documents, but they do not automatically make
# a response better.

# LlamaIndex Q3 Observation Template
#
# Suggested difficult query: "What is BrightLeaf's parental leave policy for
# remote contractors in Canada?"
# This turned out to be a good stress test. The pipeline retrieved the benefits,
# remote work, and overview documents, then answered that the documents do not
# specify a parental-leave policy for remote contractors in Canada. That was a
# reasonable outcome because the available context only covered eligible
# full-time employees and U.S.-based remote staff. To handle this kind of query
# better, I would add more HR policy documents or metadata that distinguishes
# employee type, region, and contract status.

# LlamaIndex Q4
#
# A faithfulness score of 1.0 means the evaluator judged the answer to be fully
# supported by the retrieved context. A score of 0.0 would indicate the answer
# was not supported and likely hallucinated or contradicted by the evidence.
#
# Relevancy measures whether the response and retrieved context are actually on
# topic for the question being asked. That differs from faithfulness because an
# answer can be faithful to the retrieved text but still be irrelevant if the
# system pulled the wrong chunks.
#
# In my run, the scores did not change: both the employee-benefits query and the
# unsupported visa-sponsorship query received 1.0 for faithfulness and 1.0 for
# relevancy. That happened because the second answer correctly said the policy
# was not present in the documents, so it stayed grounded and on-topic rather
# than hallucinating a made-up rule.
#
# LLM-as-a-judge means another model reviews the query, retrieved evidence, and
# answer to score qualities like support and relevance. It is used in RAG
# evaluation because many open-ended answers do not have a single simple
# ground-truth string that can be checked with exact-match accuracy.


def import_llamaindex_dependencies():
    try:
        from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
        from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
        from llama_index.embeddings.openai import OpenAIEmbedding
    except ImportError as exc:
        print(
            "Skipping LlamaIndex section because required packages are missing.\n"
            'Install at least: pip install pypdf "llama-index-core==0.14.10" '
            "llama-index-embeddings-openai"
        )
        return None

    try:
        from llama_index.llms.openai import OpenAI as LlamaOpenAI
    except ImportError:
        print(
            "Skipping LlamaIndex section because llama-index's OpenAI LLM adapter "
            "is missing.\nInstall: pip install llama-index-llms-openai"
        )
        return None

    return (
        Settings,
        SimpleDirectoryReader,
        VectorStoreIndex,
        FaithfulnessEvaluator,
        RelevancyEvaluator,
        OpenAIEmbedding,
        LlamaOpenAI,
    )


def find_brightleaf_pdf_dir() -> Path | None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    candidate_dirs = [
        script_dir / ".." / ".." / "06_AI_augmentation" / "brightleaf_pdfs",
        repo_root / "06_AI_augmentation" / "brightleaf_pdfs",
        repo_root.parent / "06_AI_augmentation" / "brightleaf_pdfs",
        Path.cwd() / "06_AI_augmentation" / "brightleaf_pdfs",
        Path.cwd() / ".." / "06_AI_augmentation" / "brightleaf_pdfs",
        Path.cwd() / ".." / ".." / "06_AI_augmentation" / "brightleaf_pdfs",
    ]

    for candidate in candidate_dirs:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_dir():
            return resolved

    search_root = repo_root.parent
    skip_dirs = {".git", ".venv", "node_modules", "__pycache__"}

    for current_root, dirs, _ in os.walk(search_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        if "brightleaf_pdfs" in dirs:
            return Path(current_root) / "brightleaf_pdfs"

    return None


def build_brightleaf_index():
    imports = import_llamaindex_dependencies()
    if imports is None:
        return None

    (
        Settings,
        SimpleDirectoryReader,
        VectorStoreIndex,
        FaithfulnessEvaluator,
        RelevancyEvaluator,
        OpenAIEmbedding,
        LlamaOpenAI,
    ) = imports

    pdf_dir = find_brightleaf_pdf_dir()
    if pdf_dir is None:
        print(
            "Skipping LlamaIndex section because no local brightleaf_pdfs directory "
            "was found near this repo."
        )
        return None

    print(f"Using BrightLeaf PDF directory: {pdf_dir}")

    Settings.llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    try:
        documents = SimpleDirectoryReader(str(pdf_dir)).load_data()
    except Exception as exc:
        print(f"Could not load BrightLeaf PDFs with SimpleDirectoryReader: {exc}")
        print(
            "If PDF loading fails, install any missing reader dependency such as "
            "llama-index-readers-file."
        )
        return None

    try:
        index = VectorStoreIndex.from_documents(documents)
    except Exception as exc:
        print(f"Could not build the BrightLeaf index: {exc}")
        return None

    return {
        "index": index,
        "FaithfulnessEvaluator": FaithfulnessEvaluator,
        "RelevancyEvaluator": RelevancyEvaluator,
        "LlamaOpenAI": LlamaOpenAI,
    }


def ask_and_print(index, question: str, similarity_top_k: int):
    query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
    response = query_engine.query(question)
    print(f"Question: {question}")
    print(f"Answer: {str(response).strip()}")
    print_source_nodes(response.source_nodes, limit=len(response.source_nodes))
    print()
    return response


def run_llamaindex_section() -> None:
    show_section("LLAMAINDEX")
    built = build_brightleaf_index()
    if built is None:
        print()
        return

    index = built["index"]
    FaithfulnessEvaluator = built["FaithfulnessEvaluator"]
    RelevancyEvaluator = built["RelevancyEvaluator"]
    LlamaOpenAI = built["LlamaOpenAI"]

    questions = [
        "What employee benefits does BrightLeaf offer?",
        "What are BrightLeaf's security policies?",
    ]

    print("LlamaIndex Q1")
    responses = {}
    for question in questions:
        responses[question] = ask_and_print(index, question, similarity_top_k=3)

    print("LlamaIndex Q2")
    comparison_query = questions[0]
    print("Run with similarity_top_k=1")
    ask_and_print(index, comparison_query, similarity_top_k=1)
    print("Run with similarity_top_k=5")
    ask_and_print(index, comparison_query, similarity_top_k=5)

    print("LlamaIndex Q3")
    hard_query = "What is BrightLeaf's parental leave policy for remote contractors in Canada?"
    ask_and_print(index, hard_query, similarity_top_k=3)

    print("LlamaIndex Q4")
    judge_llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0)
    faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
    relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

    eval_query_1 = "What employee benefits does BrightLeaf offer?"
    eval_response_1 = responses.get(eval_query_1) or ask_and_print(
        index, eval_query_1, similarity_top_k=3
    )
    faithfulness_result_1 = faithfulness_evaluator.evaluate_response(
        response=eval_response_1
    )
    relevancy_result_1 = relevancy_evaluator.evaluate_response(
        query=eval_query_1, response=eval_response_1
    )

    print(f"Query 1 faithfulness score: {safe_score(faithfulness_result_1.score)}")
    print(f"Query 1 faithfulness passing: {faithfulness_result_1.passing}")
    print(f"Query 1 relevancy score: {safe_score(relevancy_result_1.score)}")
    print(f"Query 1 relevancy passing: {relevancy_result_1.passing}")
    print()

    eval_query_2 = "What is BrightLeaf's policy for student visa sponsorship?"
    eval_response_2 = ask_and_print(index, eval_query_2, similarity_top_k=3)
    faithfulness_result_2 = faithfulness_evaluator.evaluate_response(
        response=eval_response_2
    )
    relevancy_result_2 = relevancy_evaluator.evaluate_response(
        query=eval_query_2, response=eval_response_2
    )

    print(f"Query 2 faithfulness score: {safe_score(faithfulness_result_2.score)}")
    print(f"Query 2 faithfulness passing: {faithfulness_result_2.passing}")
    print(f"Query 2 relevancy score: {safe_score(relevancy_result_2.score)}")
    print(f"Query 2 relevancy passing: {relevancy_result_2.passing}")
    print()


def main() -> None:
    run_concepts_section()
    run_keyword_section()
    run_semantic_concepts_section()
    run_llamaindex_section()


if __name__ == "__main__":
    main()
