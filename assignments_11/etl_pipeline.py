"""Week 11 capstone: Prefect ETL pipeline for weather classification.

Video link: https://youtu.be/w6XEVrgQIs4
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from openai import OpenAI
from prefect import flow, task


ACCOUNT_URL = "https://nataliactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
LATITUDE = 35.2271
LONGITUDE = -80.8431
MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)
VALID_LABELS = {"good", "marginal", "bad"}


def build_weather_url(latitude: float, longitude: float) -> str:
    """Build the Open-Meteo API URL."""
    return (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&hourly=temperature_2m,precipitation"
        "&forecast_days=7"
    )


def reshape_hourly_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Open-Meteo hourly arrays into per-hour records."""
    hourly = payload["hourly"]
    return [
        {
            "time": time_value,
            "temperature_2m": temp_value,
            "precipitation": precip_value,
        }
        for time_value, temp_value, precip_value in zip(
            hourly["time"],
            hourly["temperature_2m"],
            hourly["precipitation"],
            strict=True,
        )
    ]


def classify_record(client: OpenAI, record: dict[str, Any]) -> str:
    """Classify a single weather record with the OpenAI API."""
    user_message = (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    label = response.choices[0].message.content.strip().lower()
    return label if label in VALID_LABELS else "unknown"


@task(retries=2, retry_delay_seconds=10)
def extract_weather_data() -> dict[str, Any]:
    """Extract hourly weather data from Open-Meteo."""
    weather_url = build_weather_url(LATITUDE, LONGITUDE)
    response = requests.get(weather_url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    print(f"Extracted weather data from {weather_url}")
    return payload


@task
def transform_weather_data(raw_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Transform raw hourly weather data and classify the first 24 records."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from your environment or .env file.")

    llm_client = OpenAI(api_key=api_key)
    records = reshape_hourly_records(raw_payload)
    records_to_process = records[:24]
    print(f"Reshaped {len(records)} hourly records; classifying the first 24.")

    enriched_records: list[dict[str, Any]] = []
    for index, record in enumerate(records_to_process, start=1):
        enriched_record = dict(record)
        enriched_record["conditions"] = classify_record(llm_client, record)
        enriched_records.append(enriched_record)

        if index % 6 == 0:
            print(f"Processed {index} records...")

    return enriched_records


@task
def load_enriched_weather(enriched_records: list[dict[str, Any]]) -> str:
    """Load enriched weather records into Azure Blob Storage."""
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    blob_service_client = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)
    container_client = blob_service_client.get_container_client(CONTAINER)

    today = date.today().isoformat()
    blob_path = f"final/{today}/weather_etl.json"
    payload_bytes = json.dumps(enriched_records, indent=2).encode("utf-8")

    container_client.upload_blob(
        name=blob_path,
        data=payload_bytes,
        overwrite=True,
    )
    print(f"Uploaded {len(payload_bytes)} bytes to {blob_path}")
    return blob_path


@flow(log_prints=True)
def run_weather_etl() -> str:
    """Run the full ETL pipeline."""
    raw_payload = extract_weather_data()
    enriched_records = transform_weather_data(raw_payload)
    final_blob_path = load_enriched_weather(enriched_records)
    print(f"Pipeline completed successfully. Final blob path: {final_blob_path}")
    return final_blob_path


if __name__ == "__main__":
    run_weather_etl()
