"""Week 9 warmup answers and Blob Storage helper functions."""


# --- Azure Authentication ---
# Q1
#
# When a local Python script uses DefaultAzureCredential, it checks a chain of
# possible credential sources and uses the first one that works. In this
# homework setup, the most common successful source is your Azure CLI login
# session. You must run `az login` first so the Azure CLI can cache an access
# token for your account. DefaultAzureCredential knows to use that login because
# AzureCliCredential is one of the credential types in its built-in chain.
#
# Q2
#
# A deployed pipeline on an Azure VM, App Service, or container should not use
# `az login` because there is no human sitting there to complete an interactive
# sign-in, and storing a personal login on deployed infrastructure is not a good
# security model. Instead, it typically uses a managed identity or another
# service principal-based credential assigned to the Azure resource. The same
# Python code can still work without changes because DefaultAzureCredential tries
# multiple credential types and automatically uses the managed identity in Azure
# instead of the Azure CLI credential it would use locally.
#
# Q3
#
# The two most likely causes are:
# 1. You are not logged in locally, or your Azure CLI session has expired. I
#    would diagnose that by running `az account show` or `az login` in the same
#    terminal to confirm the CLI credential is available.
# 2. The script environment is missing a usable credential source or is pointed
#    at the wrong Azure context. I would check the full authentication error
#    message to see which credential types DefaultAzureCredential attempted, then
#    verify environment variables, subscription selection, and whether the Azure
#    resource being accessed actually allows my identity.


# --- Blob Storage ---
# Q1
#
# Azure Blob Storage has a three-level hierarchy: storage account, container,
# and blob. The storage account is the top-level Azure resource that owns the
# service. A container is like a named bucket or folder inside that account that
# groups related files. A blob is the actual file object stored inside the
# container. Using a filesystem analogy, the storage account is like the disk,
# the container is like a top-level folder, and the blob is like an individual
# file inside that folder.
#
# Q2
#
# A REST API returns a JSON payload each hour. You need to store the raw
# responses for reprocessing later.
# Use Blob Storage because you want to keep raw semi-structured files exactly as
# they arrived, not query them like normalized rows.
#
# Your pipeline produces a table of 50 million customer transactions that your
# analytics team queries by date range and customer ID every day.
# Use a relational database because the primary need is efficient filtering,
# indexing, and repeated structured queries over tabular data.
#
# A computer vision model produces image embeddings as NumPy arrays. You need to
# save them between pipeline runs.
# Use Blob Storage because embeddings are file-like binary artifacts that are
# easy to persist as objects without needing relational query behavior.


def list_container(container_client):
    """Print the name and size of each blob in a container."""
    for blob in container_client.list_blobs():
        print(f"{blob.name}: {blob.size} bytes")


def upload_text(container_client, blob_name, text):
    """Upload a UTF-8 string to Blob Storage, overwriting any existing blob."""
    container_client.upload_blob(
        name=blob_name,
        data=text.encode("utf-8"),
        overwrite=True,
    )
