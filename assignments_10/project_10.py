"""Week 10 project: LLM transform pipeline for weather records.

Video link: https://youtu.be/4QwAhu601o4

This pipeline works, but it is not an especially strong use case for an LLM.
The classification is based on only two numeric inputs, so deterministic rules
could classify the records more cheaply, faster, and with fully predictable
behavior. An LLM does give you flexible judgment around borderline cases without
hard-coding thresholds, but you lose consistency and add cost and latency. A
rule-based approach would be easier to audit, while the LLM approach is more
adaptable if the classification criteria need to become more qualitative later.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from openai import OpenAI


ACCOUNT_URL = "https://nataliactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)
VALID_LABELS = {"good", "marginal", "bad"}
OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "first_10_records.json"
FALLBACK_PATH = Path("assignments/resources/weather_raw.json")


def reshape_hourly_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Open-Meteo parallel hourly arrays into per-hour dictionaries."""
    hourly = payload["hourly"]
    times = hourly["time"]
    temperatures = hourly["temperature_2m"]
    precipitation = hourly["precipitation"]

    return [
        {
            "time": time_value,
            "temperature_2m": temp_value,
            "precipitation": precip_value,
        }
        for time_value, temp_value, precip_value in zip(
            times,
            temperatures,
            precipitation,
            strict=True,
        )
    ]


def load_weather_payload(container_client) -> tuple[dict[str, Any], str]:
    """Load today's Week 9 weather blob, falling back to the local resource file."""
    today = date.today().isoformat()
    blob_path = f"raw/{today}/weather.json"

    try:
        blob_bytes = container_client.download_blob(blob_path).readall()
        print(f"Loaded source blob: {blob_path}")
        return json.loads(blob_bytes.decode("utf-8")), blob_path
    except ResourceNotFoundError:
        print(f"Blob {blob_path} not found. Falling back to {FALLBACK_PATH}.")
        with FALLBACK_PATH.open("r", encoding="utf-8") as fallback_file:
            return json.load(fallback_file), str(FALLBACK_PATH)


def classify_record(client: OpenAI, record: dict[str, Any]) -> str:
    """Classify one hourly weather record with the OpenAI API."""
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


def main() -> None:
    """Run the read-transform-write pipeline."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from your environment or .env file.")

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    blob_service_client = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)
    container_client = blob_service_client.get_container_client(CONTAINER)
    llm_client = OpenAI(api_key=api_key)

    payload, source_path = load_weather_payload(container_client)
    records = reshape_hourly_records(payload)
    records_to_process = records[:24]
    print(f"Loaded {len(records)} hourly records from {source_path}.")
    print(f"Processing the first {len(records_to_process)} records.")

    enriched_records: list[dict[str, Any]] = []
    for index, record in enumerate(records_to_process, start=1):
        enriched_record = dict(record)
        enriched_record["conditions"] = classify_record(llm_client, record)
        enriched_records.append(enriched_record)

        if index % 6 == 0:
            print(f"Processed {index} records...")

    today = date.today().isoformat()
    processed_blob_path = f"processed/{today}/weather_classified.json"
    payload_bytes = json.dumps(enriched_records, indent=2).encode("utf-8")
    container_client.upload_blob(
        name=processed_blob_path,
        data=payload_bytes,
        overwrite=True,
    )
    print(f"Uploaded {len(payload_bytes)} bytes to {processed_blob_path}")

    downloaded_processed_bytes = container_client.download_blob(processed_blob_path).readall()
    processed_records = json.loads(downloaded_processed_bytes.decode("utf-8"))
    df = pd.DataFrame(processed_records)

    print("\nCondition counts:")
    print(df["conditions"].value_counts())
    print("\nFirst 5 rows:")
    print(df.head())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(enriched_records[:10], output_file, indent=2)
    print(f"\nSaved first 10 enriched records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
