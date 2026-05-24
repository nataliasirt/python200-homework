from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, ToolCallingAgent, tool

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
BIKE_DATA_PATH = SCRIPT_DIR / "bike_commute.csv"
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

client = OpenAI(api_key=api_key)


def show_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def ensure_bike_commute_csv() -> Path:
    if BIKE_DATA_PATH.exists():
        return BIKE_DATA_PATH

    sample_df = pd.DataFrame(
        [
            {
                "route_name": "River Loop",
                "duration_min": 22,
                "distance_km": 7.1,
                "avg_speed_kmh": 19.4,
                "avg_heart_rate": 129,
                "avg_traffic_density": 31,
            },
            {
                "route_name": "Downtown Dash",
                "duration_min": 28,
                "distance_km": 7.4,
                "avg_speed_kmh": 15.8,
                "avg_heart_rate": 141,
                "avg_traffic_density": 76,
            },
            {
                "route_name": "Lakefront",
                "duration_min": 35,
                "distance_km": 12.3,
                "avg_speed_kmh": 21.1,
                "avg_heart_rate": 136,
                "avg_traffic_density": 22,
            },
            {
                "route_name": "Campus Cut",
                "duration_min": 18,
                "distance_km": 5.2,
                "avg_speed_kmh": 17.3,
                "avg_heart_rate": 133,
                "avg_traffic_density": 44,
            },
            {
                "route_name": "Market Street",
                "duration_min": 31,
                "distance_km": 8.1,
                "avg_speed_kmh": 15.1,
                "avg_heart_rate": 145,
                "avg_traffic_density": 81,
            },
            {
                "route_name": "Greenway",
                "duration_min": 41,
                "distance_km": 14.0,
                "avg_speed_kmh": 20.5,
                "avg_heart_rate": 138,
                "avg_traffic_density": 27,
            },
            {
                "route_name": "Station Run",
                "duration_min": 24,
                "distance_km": 6.2,
                "avg_speed_kmh": 16.2,
                "avg_heart_rate": 140,
                "avg_traffic_density": 68,
            },
            {
                "route_name": "Museum Mile",
                "duration_min": 26,
                "distance_km": 6.6,
                "avg_speed_kmh": 16.0,
                "avg_heart_rate": 142,
                "avg_traffic_density": 71,
            },
            {
                "route_name": "Neighborhood Spin",
                "duration_min": 33,
                "distance_km": 10.2,
                "avg_speed_kmh": 18.6,
                "avg_heart_rate": 134,
                "avg_traffic_density": 39,
            },
            {
                "route_name": "Bridge Route",
                "duration_min": 29,
                "distance_km": 8.0,
                "avg_speed_kmh": 16.5,
                "avg_heart_rate": 143,
                "avg_traffic_density": 63,
            },
        ]
    )
    sample_df.to_csv(BIKE_DATA_PATH, index=False)
    return BIKE_DATA_PATH


def tool_call_to_message(tool_call: Any) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


def stringify_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=2, default=str)


def get_current_time() -> str:
    """Return the current local time as a readable string."""
    return datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")


# --- Lesson 02 ---

# Q1
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Return the current local time as a readable string.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit and return a formatted string.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "Temperature in degrees Celsius.",
                }
            },
            "required": ["celsius"],
        },
    },
}


def run_agent(user_query: str, tools: list[dict[str, Any]], max_rounds: int = 4) -> str:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a concise assistant. Use a tool only when it is genuinely needed. "
                "If you can answer directly, do so."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    for round_number in range(1, max_rounds + 1):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=messages,
            tools=tools,
        )
        assistant_message = response.choices[0].message
        print(f"Agent round {round_number} tool calls: {len(assistant_message.tool_calls or [])}")

        assistant_payload = {
            "role": "assistant",
            "content": assistant_message.content or "",
        }
        if assistant_message.tool_calls:
            assistant_payload["tool_calls"] = [
                tool_call_to_message(tool_call)
                for tool_call in assistant_message.tool_calls
            ]
        messages.append(assistant_payload)

        if not assistant_message.tool_calls:
            return assistant_message.content or ""

        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")

            if function_name == "get_current_time":
                tool_result = get_current_time()
            elif function_name == "celsius_to_fahrenheit":
                tool_result = celsius_to_fahrenheit(**arguments)
            else:
                tool_result = f"Unknown tool requested: {function_name}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": stringify_tool_result(tool_result),
                }
            )

    return "The agent hit the maximum number of rounds before finishing."


# --- Lesson 03 ---


class CsvManager:
    def __init__(self, base_dir: Path, outputs_dir: Path):
        self.base_dir = base_dir
        self.outputs_dir = outputs_dir
        self.df: pd.DataFrame | None = None
        self.current_path: Path | None = None

    def resolve_csv_path(self, filename: str) -> Path | None:
        candidate = Path(filename)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        candidate_paths = [
            self.base_dir / filename,
            REPO_ROOT / filename,
            SCRIPT_DIR / filename,
        ]
        for path in candidate_paths:
            if path.exists():
                return path.resolve()
        return None

    def load_csv(self, filename: str) -> dict[str, Any]:
        path = self.resolve_csv_path(filename)
        if path is None:
            return {"error": f"CSV file not found: {filename}"}

        self.df = pd.read_csv(path)
        self.current_path = path
        return {
            "message": f"Loaded {path.name}",
            "path": str(path),
            "shape": list(self.df.shape),
            "columns": self.df.columns.tolist(),
        }

    def list_columns(self) -> dict[str, Any]:
        if self.df is None:
            return {"error": "No CSV is loaded."}
        return {"columns": self.df.columns.tolist()}

    def peek_data(self, n: int = 5) -> dict[str, Any]:
        if self.df is None:
            return {"error": "No CSV is loaded."}
        return {"rows": self.df.head(n).to_dict(orient="records")}

    def summarize_column(self, column: str) -> dict[str, Any]:
        if self.df is None:
            return {"error": "No CSV is loaded."}
        if column not in self.df.columns:
            return {"error": f"Column not found: {column}"}

        summary = self.df[column].describe(include="all")
        clean_summary = {
            key: (
                value.item()
                if hasattr(value, "item")
                else value
            )
            for key, value in summary.to_dict().items()
        }
        return {"column": column, "summary": clean_summary}

    def plot_scatter(self, x_col: str, y_col: str, output_path: str = "outputs/scatter_plot.png") -> dict[str, Any]:
        if self.df is None:
            return {"error": "No CSV is loaded."}
        if x_col not in self.df.columns or y_col not in self.df.columns:
            return {"error": "One or both columns were not found in the loaded DataFrame."}

        final_path = Path(output_path)
        if not final_path.is_absolute():
            final_path = SCRIPT_DIR / final_path
        final_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.scatter(self.df[x_col], self.df[y_col], color="blue", alpha=0.8)
        plt.title(f"{y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(final_path, dpi=200)
        plt.close()

        return {
            "message": "Scatter plot created with the built-in plotting tool.",
            "path": str(final_path),
            "color_used": "blue",
        }

    def compute_correlation(self, col1: str, col2: str) -> dict[str, Any]:
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        if self.df is None:
            return {"error": "No CSV is loaded."}
        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {"error": f"One or both columns were not found: {col1}, {col2}"}

        valid_data = self.df[[col1, col2]].dropna()
        if len(valid_data) < 2:
            return {"error": "Not enough non-null rows to compute a correlation."}

        pearson_r, p_value = pearsonr(valid_data[col1], valid_data[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(pearson_r), 4),
            "p_value": round(float(p_value), 4),
        }


csv_manager = CsvManager(base_dir=SCRIPT_DIR, outputs_dir=OUTPUTS_DIR)

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file into memory so the agent can inspect and analyze it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path or filename for the CSV to load.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_columns",
            "description": "List all columns in the currently loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "peek_data",
            "description": "Preview the first few rows of the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of rows to preview.",
                        "default": 5,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_column",
            "description": "Return summary statistics for a column in the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to summarize.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_scatter",
            "description": "Create a scatter plot from two columns and save it to disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_col": {
                        "type": "string",
                        "description": "Column for the x-axis.",
                    },
                    "y_col": {
                        "type": "string",
                        "description": "Column for the y-axis.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "File path for the saved plot.",
                        "default": "outputs/scatter_plot.png",
                    },
                },
                "required": ["x_col", "y_col"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation and p-value for two numeric columns in the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "The first numeric column.",
                    },
                    "col2": {
                        "type": "string",
                        "description": "The second numeric column.",
                    },
                },
                "required": ["col1", "col2"],
            },
        },
    },
]

node_tools = {
    "load_csv": csv_manager.load_csv,
    "list_columns": csv_manager.list_columns,
    "peek_data": csv_manager.peek_data,
    "summarize_column": csv_manager.summarize_column,
    "plot_scatter": csv_manager.plot_scatter,
    "compute_correlation": csv_manager.compute_correlation,
}

SYSTEM_PROMPT = """
You are a CSV analysis assistant using a ReAct-style loop.
Use tools to load the file, inspect columns, summarize values, compute correlations,
and make plots. Do not invent data you have not observed.
When you have enough evidence, answer clearly and briefly.
""".strip()


def run_agent_cycle(messages: list[dict[str, Any]], user_input: str, max_rounds: int = 6) -> str:
    messages.append({"role": "user", "content": user_input})

    for round_number in range(1, max_rounds + 1):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=messages,
            tools=tools_schema,
        )
        assistant_message = response.choices[0].message
        print(f"CSV agent round {round_number} tool calls: {len(assistant_message.tool_calls or [])}")

        assistant_payload = {
            "role": "assistant",
            "content": assistant_message.content or "",
        }
        if assistant_message.tool_calls:
            assistant_payload["tool_calls"] = [
                tool_call_to_message(tool_call)
                for tool_call in assistant_message.tool_calls
            ]
        messages.append(assistant_payload)

        if not assistant_message.tool_calls:
            return assistant_message.content or ""

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")
            tool_result = node_tools[tool_name](**tool_args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": stringify_tool_result(tool_result),
                }
            )

    return "The agent hit the tool-round limit before finishing."


# --- Lesson 04 ---


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file into the shared CSV manager.

    Args:
        filename: The CSV filename or path to load.

    Returns:
        A dictionary describing the loaded file, including its shape and columns.
    """
    return csv_manager.load_csv(filename)


@tool
def list_columns() -> dict:
    """List all columns in the currently loaded CSV.

    Returns:
        A dictionary containing the current column names.
    """
    return csv_manager.list_columns()


@tool
def peek_data(n: int = 5) -> dict:
    """Preview the first few rows of the currently loaded CSV.

    Args:
        n: The number of rows to return.

    Returns:
        A dictionary containing the first rows as records.
    """
    return csv_manager.peek_data(n=n)


@tool
def summarize_column(column: str) -> dict:
    """Summarize one column from the currently loaded CSV.

    Args:
        column: The column to summarize.

    Returns:
        A dictionary containing descriptive statistics for the requested column.
    """
    return csv_manager.summarize_column(column)


@tool
def plot_scatter(x_col: str, y_col: str, output_path: str = "outputs/scatter_plot.png") -> dict:
    """Create a scatter plot using the built-in plotting tool.

    Args:
        x_col: Column name for the x-axis.
        y_col: Column name for the y-axis.
        output_path: Where to save the PNG file.

    Returns:
        A dictionary describing the saved plot.
    """
    return csv_manager.plot_scatter(x_col=x_col, y_col=y_col, output_path=output_path)


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation and p-value for two numeric columns.

    Args:
        col1: The first numeric column name.
        col2: The second numeric column name.

    Returns:
        A dictionary containing the two column names, Pearson r, and p-value.
    """
    return csv_manager.compute_correlation(col1, col2)


SMOLAGENTS_SYSTEM_PROMPT = """
You are a student-friendly data assistant.
Use tools first when they can solve the task directly.
Write Python code when tool behavior is too limited, especially for custom plots.
The built-in plot_scatter tool only creates blue dots. If the user asks for a
specific color or other styling that the tool does not expose, inspect csv_manager.df
and write matplotlib code yourself instead of calling plot_scatter.
Save any generated plot files inside the outputs directory.
""".strip()


def run_lesson_02() -> None:
    show_section("Lesson 02: Tool Definitions and the ReAct Loop")

    print("Q1 schema for celsius_to_fahrenheit:")
    print(json.dumps(celsius_to_fahrenheit_schema, indent=2))
    print(celsius_to_fahrenheit(0))
    print(celsius_to_fahrenheit(100))
    print(celsius_to_fahrenheit(-40))

    # Q2 prediction:
    # I do not expect a tool call here because the only available tool is get_current_time,
    # which has nothing to do with temperature conversion. The model can answer the math
    # directly, so I expect exactly one API call.
    print("\nQ2 result:")
    q2_result = run_agent("Convert 100 degrees Celsius to Fahrenheit", tools=[get_current_time_schema])
    print(q2_result)
    # After running it, compare the printed round count to the prediction above.

    print("\nQ3 results:")
    response_a = run_agent(
        "What is 37 degrees Celsius in Fahrenheit?",
        tools=[get_current_time_schema, celsius_to_fahrenheit_schema],
    )
    print("Response A:", response_a)
    # A tool should be called here because the query directly matches the temperature-conversion tool.

    response_b = run_agent(
        "What is the boiling point of water in plain English?",
        tools=[get_current_time_schema, celsius_to_fahrenheit_schema],
    )
    print("Response B:", response_b)
    # A tool may not be needed here because the model can answer from general knowledge without calculation.


def run_lesson_03() -> list[dict[str, Any]]:
    show_section("Lesson 03: Multi-Tool Agent")
    ensure_bike_commute_csv()

    # Q4 adds compute_correlation to both the tool schema list and the node_tools dispatch table above.
    print("Q4 tool names:")
    print([tool["function"]["name"] for tool in tools_schema])

    # Q5
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    result = run_agent_cycle(
        messages,
        "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.",
    )
    print("Q5 final response:")
    print(result)

    # Q6 role explanation:
    # system: the standing instructions that define the agent's behavior and tool-use policy
    # user: the human request that starts the current task
    # assistant: the model's reasoning step or its final natural-language response
    # tool: the observation returned after a requested tool actually runs
    print("\nQ6 full messages list:")
    print(json.dumps(messages, indent=2, default=str))
    return messages


def run_lesson_04() -> None:
    show_section("Lesson 04: smolagents")
    ensure_bike_commute_csv()
    csv_manager.load_csv("bike_commute.csv")

    print("Q7 compute_correlation.description:")
    print(compute_correlation.description)
    # smolagents builds a tool description automatically from the function signature and docstring,
    # while the manual JSON schema in Q4 required every field to be written explicitly.
    # To produce a good auto-generated description, the developer still needs a clear function name,
    # accurate parameter names and types, and a docstring that explains purpose, inputs, and outputs.

    model = OpenAIServerModel(api_key=api_key, model_id=MODEL_NAME)
    tools = [load_csv, list_columns, peek_data, summarize_column, plot_scatter, compute_correlation]

    tool_agent = ToolCallingAgent(
        tools=tools,
        model=model,
        instructions=SMOLAGENTS_SYSTEM_PROMPT,
        max_steps=6,
    )
    code_agent = CodeAgent(
        tools=tools,
        model=model,
        instructions=SMOLAGENTS_SYSTEM_PROMPT,
        additional_authorized_imports=["pandas", "matplotlib", "matplotlib.pyplot", "pathlib"],
        max_steps=6,
    )

    prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

    response_tool = tool_agent.run(prompt)
    response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})

    print("Q8 ToolCallingAgent response:")
    print(response_tool)
    print("Q8 CodeAgent response:")
    print(response_code)

    # In this run, the ToolCallingAgent did not successfully execute a green-dot plot through the
    # tool interface. Instead, it returned a Python example the student could run manually.
    # The CodeAgent wrote and executed matplotlib code, so it actually created the green-dot plot.
    # This shows that ToolCallingAgent is better when the task fits the available tool contracts,
    # while CodeAgent is more useful when the request needs custom logic beyond the tool interface.

    # Q9
    # A ToolCallingAgent would be a better choice for a tightly controlled workflow like
    # checking account balances, booking an appointment, or querying a small analytics API,
    # because the task can be expressed as safe, pre-approved tool calls with predictable inputs.
    #
    # One meaningful CodeAgent risk is that it generates and executes code, which can fail in
    # unexpected ways, consume local resources, or produce side effects that a strict tool-only
    # agent would not be able to trigger.


def main() -> None:
    run_lesson_02()
    run_lesson_03()
    run_lesson_04()


if __name__ == "__main__":
    main()
