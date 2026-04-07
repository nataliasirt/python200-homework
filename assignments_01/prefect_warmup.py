"""
Prefect Pipeline Warmup Exercise for Assignment 01

This file rebuilds the data pipeline from Pipeline Question 1 using Prefect
tasks and flows instead of plain Python functions.
"""

import pandas as pd
import numpy as np
from prefect import task, flow


# --- Prefect Tasks ---

@task
def create_series(arr):
    """Takes a NumPy array and returns a pandas Series with the name 'values'."""
    return pd.Series(arr, name="values")


@task
def clean_data(series):
    """Takes a Series, removes NaN values using .dropna(), and returns the cleaned Series."""
    return series.dropna()


@task
def summarize_data(series):
    """
    Takes a Series and returns a dictionary with mean, median, std, and mode.
    """
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary


# --- Prefect Flow ---

@flow
def data_pipeline_flow():
    """
    Prefect flow that chains the three tasks together:
    create_series -> clean_data -> summarize_data
    """
    # Create the array with missing values
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    
    # Call the tasks in sequence
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    
    return summary


# --- Main Entry Point ---

if __name__ == "__main__":
    print("=" * 50)
    print("Pipeline Question 2: Prefect Workflow")
    print("=" * 50)
    
    result = data_pipeline_flow()
    
    print(f"\nPipeline Results:")
    print(f"Mean: {result['mean']}")
    print(f"Median: {result['median']}")
    print(f"Std: {result['std']}")
    print(f"Mode: {result['mode']}")
    
    print("\n" + "=" * 50)
    
    # Comment block answering the questions
    print("\nComment: Why Prefect might be overkill for this simple pipeline:")
    print("""
Prefect adds significant overhead for this simple use case:
1. The pipeline has only three small functions processing a handful of numbers
2. There's no actual computation bottleneck or need for scheduling
3. No error recovery, monitoring, or distributed processing is needed
4. Prefect's benefits (task orchestration, retries, logging, UI) aren't utilized

For this script, the plain Python version from Q1 is simpler and faster.
    """)
    
    print("\nComment: Realistic scenarios where Prefect would be valuable:")
    print("""
Even with simple pipeline logic, Prefect becomes useful when:
1. Long-running workflows: Pipelines that take hours/days benefit from retries
   and checkpointing to avoid recomputing from scratch on failure

2. Scheduled execution: Automating daily/weekly data processing where you
   need reliable scheduling, email alerts on failures, and execution history

3. Resource constraints: Running on multiple machines or cloud environments
   where task distribution and monitoring becomes complex

4. Monitoring & observability: Production systems need logging, alerting,
   dashboards, and audit trails - all provided by Prefect

5. Data dependencies: Complex workflows with many interdependent steps
   requiring robust state management and retry logic

6. Team collaboration: When multiple engineers maintain workflows,
   Prefect's UI and API make debugging and understanding data lineage easier
    """)
    
    print("=" * 50)
