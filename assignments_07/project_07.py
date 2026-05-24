from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
DATA_PATH = REPO_ROOT / "assignments_01" / "outputs" / "merged_happiness.csv"
RAW_DATA_DIR = REPO_ROOT / "assignments" / "resources" / "happiness_project"
MODEL_NAME = "gpt-4o-mini"

os.chdir(SCRIPT_DIR)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

if load_dotenv(REPO_ROOT / ".env"):
    print("Loaded environment variables from .env")
else:
    print("Warning: could not load .env file")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("OPENAI_API_KEY is not set. Add it to the repo .env file.")

df: pd.DataFrame | None = None


def to_native(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def ensure_happiness_dataframe() -> pd.DataFrame:
    global df

    if df is None:
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
        else:
            df = merge_happiness_files()
        if "happiness_score" in df.columns and "ladder_score" in df.columns:
            df["happiness_score"] = df["happiness_score"].fillna(df["ladder_score"])
    return df


def merge_happiness_files() -> pd.DataFrame:
    yearly_frames: list[pd.DataFrame] = []

    for year in range(2015, 2025):
        file_path = RAW_DATA_DIR / f"world_happiness_{year}.csv"
        if not file_path.exists():
            continue

        yearly_df = pd.read_csv(file_path, sep=";", decimal=",")
        yearly_df.columns = (
            yearly_df.columns.str.strip().str.lower().str.replace(" ", "_")
        )
        yearly_df["year"] = year
        yearly_frames.append(yearly_df)

    if not yearly_frames:
        raise FileNotFoundError("No yearly World Happiness CSV files were found.")

    merged_df = pd.concat(yearly_frames, ignore_index=True)
    return merged_df


@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    This tool first tries to load the merged CSV created in Week 1. If that file
    does not exist, it falls back to loading the yearly source files from the
    happiness project resource directory and merging them into one DataFrame.
    The loaded DataFrame is stored in the shared global variable `df`.

    Returns:
        A dictionary containing the dataset shape and full list of columns.
    """
    loaded_df = ensure_happiness_dataframe()
    return {"shape": list(loaded_df.shape), "columns": loaded_df.columns.tolist()}


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Args:
        column: The name of the column to summarize.

    Returns:
        A dictionary of descriptive statistics produced by pandas describe().
        Returns an error dictionary if no data is loaded or the column is missing.
    """
    if df is None:
        return {"error": "No dataset is loaded. Run load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}

    summary = {key: to_native(value) for key, value in df[column].describe().to_dict().items()}
    return {"column": column, "summary": summary}


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: The first numeric column name.
        col2: The second numeric column name.

    Returns:
        A dictionary containing the requested column names, Pearson r, and p-value.
        Returns an error dictionary if no data is loaded, a column is missing, or
        there are not enough valid rows to compute the statistic.
    """
    if df is None:
        return {"error": "No dataset is loaded. Run load_happiness_data first."}
    if col1 not in df.columns or col2 not in df.columns:
        return {"error": f"One or both columns were not found: {col1}, {col2}"}

    valid_data = df[[col1, col2]].dropna()
    if len(valid_data) < 2:
        return {"error": "Not enough non-null rows to compute a correlation."}

    pearson_r, p_value = pearsonr(valid_data[col1], valid_data[col2])
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(pearson_r), 4),
        "p_value": round(float(p_value), 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column: The metric column to rank by.
        year: The dataset year to filter to.
        n: The number of top countries to return. Defaults to 5.

    Returns:
        A dictionary containing the requested year, column, and top-ranked countries.
        Each result row includes the country name and the requested metric value.
        Returns an error dictionary if the data is not loaded or the inputs are invalid.
    """
    if df is None:
        return {"error": "No dataset is loaded. Run load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}
    if "country" not in df.columns or "year" not in df.columns:
        return {"error": "The dataset does not include the expected country/year columns."}

    filtered = df[df["year"] == year]
    if filtered.empty:
        return {"error": f"No rows were found for year {year}."}

    top_rows = (
        filtered[["country", column]]
        .dropna()
        .sort_values(by=column, ascending=False)
        .head(n)
    )
    results = [
        {"country": row["country"], column: to_native(row[column])}
        for _, row in top_rows.iterrows()
    ]
    return {"year": year, "column": column, "top_countries": results}


SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
The canonical score column is happiness_score because it has already been normalized
across all years. Prefer happiness_score over ladder_score unless the user explicitly
asks for ladder_score.
Be concise and student-friendly in your responses.
When you create files, save them inside the outputs directory.
After you finish the task, always return the final result with final_answer instead of
stopping at an explanatory paragraph.
""".strip()


model = OpenAIServerModel(api_key=api_key, model_id=MODEL_NAME)
agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats", "pathlib"],
    max_steps=8,
)


def verify_happiness_plot() -> None:
    plot_path = OUTPUTS_DIR / "happiness_by_region.png"
    if plot_path.exists():
        print(f"Verified plot exists: {plot_path}")
    else:
        print(f"Warning: expected plot was not found at {plot_path}")


def verify_custom_plot() -> None:
    plot_path = OUTPUTS_DIR / "gdp_vs_happiness_by_year.png"
    if plot_path.exists():
        print(f"Verified custom plot exists: {plot_path}")
    else:
        print(f"Warning: expected custom plot was not found at {plot_path}")


def agent_context() -> dict[str, Any]:
    return {"df": ensure_happiness_dataframe()}


def run_guided_queries() -> None:
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False, additional_args=agent_context())
        print(response)

    verify_happiness_plot()


def run_custom_queries() -> None:
    # My query 1
    my_query_1 = (
        "Using the loaded dataframe, compute the mean happiness_score for each "
        "regional_indicator in 2024 and tell me which region is highest."
    )
    response_1 = agent.run(
        my_query_1,
        reset=False,
        additional_args=agent_context(),
    )
    print(f"\n--- My Query 1: {my_query_1} ---")
    print(response_1)
    # Comment: This should trigger both tool use and code generation because there is no dedicated region-aggregation tool.

    # My query 2
    my_query_2 = (
        "Create a scatter plot of gdp_per_capita vs happiness_score, color the points by year, "
        "and save it to outputs/gdp_vs_happiness_by_year.png."
    )
    response_2 = agent.run(
        my_query_2,
        reset=False,
        additional_args=agent_context(),
    )
    print(f"\n--- My Query 2: {my_query_2} ---")
    print(response_2)
    # Comment: This should require code generation because none of the tools create a custom multi-color scatter plot.
    verify_custom_plot()


def main() -> None:
    run_guided_queries()
    run_custom_queries()


if __name__ == "__main__":
    main()


# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?
#    The agent should explain significance by comparing the p-value to a standard threshold
#    such as 0.05. If it says the result is statistically significant when p < 0.05, that is
#    the correct interpretation for this assignment-level analysis.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
#    The most likely surprise is the plotting query: even without a custom plotting tool,
#    the CodeAgent can often write the pandas/matplotlib code on its own and save the chart
#    to disk, which feels much more flexible than the earlier tool-calling loop.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    A useful additional tool would be `filter_by_region(region: str, year: int | None = None)`.
#    It would return the matching subset or a compact summary, which would help answer questions
#    about one region over time without forcing the agent to write custom filtering code each time.
