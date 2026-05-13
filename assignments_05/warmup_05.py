from __future__ import annotations

import json
import os
import sys
try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise SystemExit(
        "python-dotenv is not installed. Install it with: pip install python-dotenv"
    ) from exc

try:
    from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
except ImportError as exc:
    raise SystemExit("openai is not installed. Install it with: pip install openai") from exc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def show_section(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_response(label: str, response) -> None:
    print(f"{label}: {response.choices[0].message.content.strip()}")


def create_chat_completion(client: OpenAI, **kwargs):
    try:
        return client.chat.completions.create(**kwargs)
    except APIConnectionError as exc:
        raise SystemExit(
            "Could not reach the OpenAI API. Check your internet connection and try again."
        ) from exc
    except RateLimitError as exc:
        raise SystemExit(
            "The request was rate-limited or your quota was exceeded. Check your API usage and billing."
        ) from exc
    except APIStatusError as exc:
        raise SystemExit(f"OpenAI API returned an error: {exc.status_code}") from exc


load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit(
        "OPENAI_API_KEY is not set. Add it to a .env file at the project root."
    )

client = OpenAI()


# --- Chat Completions API ---

# API QUESTION 1
show_section("API QUESTION 1")
response = create_chat_completion(
    client,
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?",
        }
    ],
)

print_response("Response text", response)
print(f"Model: {response.model}")
print(f"Total tokens used: {response.usage.total_tokens}")
print()


# API QUESTION 2
show_section("API QUESTION 2")
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temperature in temperatures:
    response = create_chat_completion(
        client,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    print_response(f"Temperature {temperature}", response)
    print()

# Lower temperature values usually produce more stable, repeatable wording.
# Higher temperature values usually produce more variety and more surprising phrasing.
# If I needed a consistent, reproducible output, I would use temperature=0.


# API QUESTION 3
show_section("API QUESTION 3")
response = create_chat_completion(
    client,
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library).",
        }
    ],
    n=3,
    temperature=1.0,
)

for i, choice in enumerate(response.choices, start=1):
    print(f"Choice {i}: {choice.message.content.strip()}")
print()


# API QUESTION 4
show_section("API QUESTION 4")
response = create_chat_completion(
    client,
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15,
)

print_response("Short response", response)
print(f"Finish reason: {response.choices[0].finish_reason}")
print()

# The response gets cut off early because max_tokens limits the size of the model's output.
# In a real application, max_tokens helps control cost, reduce latency, and enforce concise replies.


# --- System Messages and Personas ---

# SYSTEM QUESTION 1
show_section("SYSTEM QUESTION 1")
messages = [
    {
        "role": "system",
        "content": (
            "You are a patient, encouraging Python tutor. "
            "You always explain things simply and end with a word of encouragement."
        ),
    },
    {"role": "user", "content": "I don't understand what a list comprehension is."},
]
response = create_chat_completion(client, model="gpt-4o-mini", messages=messages)
print_response("Tutor persona", response)
print()

messages = [
    {
        "role": "system",
        "content": (
            "You are a blunt pirate captain who explains coding concepts using pirate slang "
            "and short commands."
        ),
    },
    {"role": "user", "content": "I don't understand what a list comprehension is."},
]
response = create_chat_completion(client, model="gpt-4o-mini", messages=messages)
print_response("Pirate persona", response)
print()

# The system message changed the tone, wording, and style of the explanation
# even though the user asked the exact same question both times.


# SYSTEM QUESTION 2
show_section("SYSTEM QUESTION 2")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {
        "role": "assistant",
        "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?",
    },
    {"role": "user", "content": "Can you remind me what my name is?"},
]
response = create_chat_completion(client, model="gpt-4o-mini", messages=messages)
print_response("Conversation response", response)
print()

# The model knows Jordan's name because the prior messages were included in
# the same API call, so the name was present in the context window.


# --- Prompt Engineering ---

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow.",
]


# PROMPT QUESTION 1
show_section("PROMPT QUESTION 1 - ZERO-SHOT")
for i, review in enumerate(reviews, start=1):
    response = create_chat_completion(
        client,
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify the sentiment of this review as positive, negative, or mixed. "
                    "Respond with only the sentiment label.\n"
                    f'Review: "{review}"'
                ),
            }
        ],
    )
    print_response(f"Review {i}", response)
print()


# PROMPT QUESTION 2
show_section("PROMPT QUESTION 2 - ONE-SHOT")
for i, review in enumerate(reviews, start=1):
    response = create_chat_completion(
        client,
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify the sentiment of each review as positive, negative, or mixed.\n\n"
                    'Example:\nReview: "Fast shipping but the item arrived damaged."\n'
                    "Sentiment: mixed\n\n"
                    f'Review: "{review}"\n'
                    "Sentiment:"
                ),
            }
        ],
    )
    print_response(f"Review {i}", response)
print()

# Adding one example usually makes the format more consistent because the model
# can imitate the example's label style instead of inferring the format itself.


# PROMPT QUESTION 3
show_section("PROMPT QUESTION 3 - FEW-SHOT")
for i, review in enumerate(reviews, start=1):
    response = create_chat_completion(
        client,
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify the sentiment of each review as positive, negative, or mixed.\n\n"
                    'Example 1:\nReview: "The app is easy to use and saves me time."\n'
                    "Sentiment: positive\n\n"
                    'Example 2:\nReview: "The package arrived late and two parts were missing."\n'
                    "Sentiment: negative\n\n"
                    'Example 3:\nReview: "The design looks great, but the battery life is disappointing."\n'
                    "Sentiment: mixed\n\n"
                    f'Review: "{review}"\n'
                    "Sentiment:"
                ),
            }
        ],
    )
    print_response(f"Review {i}", response)
print()

# Zero-shot is fastest when the task is simple and obvious.
# One-shot helps when you want a specific output format with minimal prompt length.
# Few-shot is best when the task is subtle or you want stronger consistency.


# PROMPT QUESTION 4
show_section("PROMPT QUESTION 4 - CHAIN OF THOUGHT")
response = create_chat_completion(
    client,
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "Solve this step by step and show your reasoning before the final answer. "
                "Label the final answer clearly.\n\n"
                "A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later "
                "takes a new job that pays $7,500 more per year than her post-raise salary. "
                "What is her final annual salary?"
            ),
        }
    ],
)
print_response("Full response", response)
print()

# Asking for step-by-step reasoning often improves accuracy because it encourages
# the model to break a multi-step problem into intermediate calculations.


# PROMPT QUESTION 5
show_section("PROMPT QUESTION 5 - STRUCTURED OUTPUT")
review = (
    "I've been using this tool for three months. It handles large datasets well, "
    "but the UI is clunky and the export options are limited."
)
response = create_chat_completion(
    client,
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "Analyze the review below and return only valid JSON with the keys "
                "sentiment, confidence, and reason. The confidence must be a float from 0 to 1.\n\n"
                f'Review: "{review}"'
            ),
        }
    ],
)
raw_json = response.choices[0].message.content.strip()
print(f"Raw response: {raw_json}")
try:
    parsed = json.loads(raw_json)
    print(f"Sentiment: {parsed['sentiment']}")
    print(f"Confidence: {parsed['confidence']}")
    print(f"Reason: {parsed['reason']}")
except json.JSONDecodeError:
    print("JSON parsing failed. Raw response shown above for debugging.")
print()


# PROMPT QUESTION 6
show_section("PROMPT QUESTION 6 - DELIMITERS")
user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the "
    "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = create_chat_completion(
    client,
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)
print_response("Instruction text", response)
print()

non_instruction_text = "The lake was calm that morning, and a thin layer of fog drifted above the water."
second_prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{non_instruction_text}```
"""
response = create_chat_completion(
    client,
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": second_prompt}],
)
print_response("Non-instruction text", response)
print()

# Delimiters help prevent confusion between the instructions for the model and
# the user-provided text that should be analyzed or transformed.


# --- Local Models with Ollama ---

# OLLAMA QUESTION 1
show_section("OLLAMA QUESTION 1")
response = create_chat_completion(
    client,
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain what a large language model is in two sentences.",
        }
    ],
)
print_response("OpenAI response", response)
print()

"""
A large language model is an AI system designed to understand and generate
human-like text, trained on vast amounts of text data to improve comprehension
and creativity. It can answer questions, write stories, and process complex
information, making it a versatile tool for tasks like customer support or
creative writing.
"""

# The Ollama response was shorter and more general, while the OpenAI response
# sounded a bit more polished and precise about how the model works.
# One advantage of running locally is privacy and zero per-request API cost.
# One disadvantage is that local models often require more setup and may be
# weaker than hosted models on quality, speed, or reasoning.
