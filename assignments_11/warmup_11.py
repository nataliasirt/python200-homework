"""Week 11 warmup answers and code."""

from prefect import get_run_logger, task


# --- Prefect Orchestration ---
# Q1
#
# A @task is a single unit of work inside a pipeline, while a @flow is the
# orchestration function that calls tasks, manages their order, and represents
# the whole workflow run. I would not decorate a pure helper like Celsius-to-
# Fahrenheit conversion with @task unless I specifically needed Prefect
# observability or retries around it, because small in-memory helpers are
# usually better left as normal Python functions to avoid unnecessary overhead.
#
# Q2
# @task(retries=3, retry_delay_seconds=30)
#
# Q3
#
# In the Prefect UI, I would open the failed flow run and then inspect the
# transform task run specifically. I would expect to find the task state, error
# message, traceback, timestamps, and logs showing exactly where the transform
# step failed. I would also confirm that extract is marked Completed and that
# load never started because Prefect stopped the downstream dependency chain
# after the transform task failed.


# --- Production Patterns ---
# Q1
#
# raise_for_status() turns an HTTP error response like 500 into an exception,
# which causes the task to fail clearly and lets Prefect record the failure and
# apply retries if configured. That is better than `if response.status_code !=
# 200: print("error")` because printing only logs a message and then the task may
# keep running with bad or missing data. With a 500 error, raise_for_status()
# stops the task immediately so downstream tasks do not run, while a print-based
# check can allow downstream tasks to run incorrectly unless you also raise an
# exception yourself.
#
# Q2
#
# overwrite=True protects you from a re-run failing just because a blob already
# exists at the target path. In this scenario, after fixing the transform bug,
# the successful rerun can safely replace the old output at
# final/{today}/weather_etl.json instead of leaving stale data or throwing a
# blob-already-exists error. Without overwrite=True, the load step could fail on
# rerun if the target blob path already exists.


@task
def log_loaded_records(records: list, blob_path: str) -> None:
    """Task stub that logs how many records were loaded."""
    logger = get_run_logger()
    logger.info("Loaded %s records from %s", len(records), blob_path)
