"""Week 9 project: Extract + Load pipeline with Open-Meteo and Azure Blob Storage.

Video link: https://youtu.be/gLwZTnqVqK4
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


ACCOUNT_URL = "https://nataliactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
LATITUDE = 35.2271
LONGITUDE = -80.8431
OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "weather_raw.json"


def build_weather_url(latitude: float, longitude: float) -> str:
    """Build the Open-Meteo forecast URL for 7 days of hourly weather data."""
    return (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&hourly=temperature_2m,precipitation"
        "&forecast_days=7"
    )


def main() -> None:
    """Run the extract + load pipeline."""
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    blob_service_client = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)
    container_client = blob_service_client.get_container_client(CONTAINER)

    weather_url = build_weather_url(LATITUDE, LONGITUDE)
    response = requests.get(weather_url, timeout=30)
    response.raise_for_status()

    payload = response.json()
    payload_bytes = json.dumps(payload).encode("utf-8")

    today = date.today().isoformat()
    blob_path = f"raw/{today}/weather.json"

    container_client.upload_blob(
        name=blob_path,
        data=payload_bytes,
        overwrite=True,
    )
    print(f"Uploaded {len(payload_bytes)} bytes to {blob_path}")

    print("\nContainer contents:")
    for blob in container_client.list_blobs():
        print(f"- {blob.name}: {blob.size} bytes")

    downloaded_bytes = container_client.download_blob(blob_path).readall()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(downloaded_bytes)

    downloaded_payload = json.loads(downloaded_bytes.decode("utf-8"))
    hourly_df = pd.DataFrame(downloaded_payload["hourly"])

    print("\nFirst 5 rows of hourly weather data:")
    print(hourly_df.head())
    print(f"\nSaved downloaded JSON to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
