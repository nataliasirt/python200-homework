"""Week 10 warmup answers and code."""


# --- LLMs as Transform ---
# Q1
#
# Parse the string "Jan 5th, 2024" into an ISO date format like "2024-01-05".
# Use deterministic code because date parsing is a fixed, structured conversion that a library can do reliably and cheaply.
#
# Classify a customer support ticket -- "my card was charged twice" -- into one of: billing, technical, or general.
# Use an LLM because the input is natural language and the task depends on semantic interpretation rather than fixed rules.
#
# Calculate the average of a list of numbers.
# Use deterministic code because arithmetic should be exact and does not need probabilistic language reasoning.
#
# Extract the company name from a freeform job title like "Sr. Data Eng @ Acme Corp (contract)".
# Use an LLM because the text is messy and variable, so semantic extraction is more robust than brittle string rules.
#
# Determine whether a product review is more than 100 words long.
# Use deterministic code because counting words is a straightforward rule-based operation.
#
# Q2
#
# The prompt "Summarize this product review in a few sentences." creates a downstream
# pipeline problem because the output shape is not constrained, so the model may
# return a different number of sentences, extra commentary, or formatting that is
# harder to parse and store consistently. In a pipeline, the prompt should define
# an exact output format so every record can be processed the same way.
#
# A better prompt would be:
# system = (
#     "Summarize the product review in exactly one sentence. "
#     "Return valid JSON with one key named summary. "
#     "Do not include markdown or any extra text."
# )
#
# Q3
#
# If 50,000 calls take 1 second each sequentially, the total runtime would be
# 50,000 seconds, which is about 13.9 hours. One practical way to handle this
# more efficiently without changing models is to process records concurrently in
# batches while respecting rate limits, so multiple API calls are in flight at
# the same time.


# --- Azure OpenAI ---
# Q1
#
# One reason an organization might use Azure OpenAI is that it can keep model
# usage inside the organization's Azure environment, which helps with enterprise
# governance, network controls, and compliance requirements. Another reason is
# that companies already using Azure can integrate OpenAI capabilities with
# existing Azure identity, billing, regional deployment, and security policies.
#
# Q2
#
# The three Azure-specific client initialization parameters are:
# 1. azure_endpoint: the base URL of your Azure OpenAI resource.
# 2. api_version: the Azure OpenAI REST API version you want to call.
# 3. azure_deployment: not passed to the client constructor itself, but the
#    deployment name is the Azure-specific model identifier you use when making
#    requests, representing the model deployment you created in Azure.
#
# Q3
#
# When using AzureOpenAI, the model parameter takes your deployment name, not a
# raw model name like "gpt-4o-mini". You find that value in the Azure OpenAI
# resource where you created the model deployment, because Azure asks you to
# deploy a model under a deployment name and that name is what the SDK call uses.
