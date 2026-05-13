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

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit(
        "OPENAI_API_KEY is not set. Add it to a .env file at the project root."
    )

client = OpenAI()

SYSTEM_PROMPT = """
You are Job Application Helper, an AI job application coach for career changers.
Your job is to help users improve resume bullet points, draft cover letter openings,
and think through application strategy in a practical, supportive way.
Stay focused on job application materials and related career communication.
Do not claim to know company-specific or industry-specific hiring norms with certainty.
If something depends on the user's field, location, or target employer, say so clearly.
Always remind the user to review and edit your output before submitting it anywhere.
Encourage the user to use their own judgment, because your suggestions may not match
their exact industry expectations or lived experience.
Keep advice concrete, honest, and free of invented achievements.
""".strip()

# I made the system prompt explicitly about career changers instead of generic job seekers
# so the assistant will consistently translate prior experience into new-role language.


def show_section(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> str | None:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except APIConnectionError:
        print("Job Application Helper: Could not reach the OpenAI API. Try again later.\n")
        return None
    except RateLimitError:
        print("Job Application Helper: The request was rate-limited or your quota was exceeded.\n")
        return None
    except APIStatusError as exc:
        print(f"Job Application Helper: The OpenAI API returned an error ({exc.status_code}).\n")
        return None


def rewrite_bullets(bullets: list[str]) -> list[dict[str, str]]:
    bullet_text = "\n".join(f"- {bullet}" for bullet in bullets)

    prompt = f"""
You are a professional resume coach helping a career changer.
Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
Use strong action verbs. Do not invent facts, numbers, metrics, timelines, or tools that are not implied by the original.
Preserve the original meaning. If the original bullet does not mention a result, do not add one.
If the original bullet says "on time," do not upgrade that to "ahead of schedule."

Return ONLY a valid JSON list. Each item should have two keys:
"original" and "improved".
Respond ONLY with valid JSON, with no markdown fences and no extra text.

Examples:
Original: "Answered customer questions"
Improved: "Resolved customer questions and concerns with clear, professional communication."

Original: "Made weekly reports for my manager"
Improved: "Prepared weekly reports for management to support visibility into ongoing work."

Bullet points:
```
{bullet_text}
```
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    raw_response = get_completion(messages, temperature=0.4)
    if raw_response is None:
        return []

    try:
        parsed = json.loads(clean_json_text(raw_response))
    except json.JSONDecodeError:
        print("Job Application Helper: Could not parse the bullet rewrite response as JSON.")
        print(raw_response)
        print()
        return []

    for i, item in enumerate(parsed, start=1):
        print(f"Bullet {i} original: {item['original']}")
        print(f"Bullet {i} improved: {item['improved']}")
        print()

    return parsed


# These sample bullets are weak because they are vague, generic, and missing outcomes.
# The model improves them by adding clearer action verbs, sharper scope, and stronger framing.


def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
You write strong cover letter opening paragraphs for career changers.
The paragraph should be 3-5 sentences: confident, specific, and free of cliches.

Here are two examples of the style and tone you should match:

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening: After seven years as a registered nurse, I've spent my career making decisions
under pressure using incomplete information - which turns out to be excellent training for
data analysis. I recently completed a data analytics program where I built dashboards
tracking patient outcomes across departments. I'm excited to bring that combination of
clinical context and technical skill to [Company]'s mission-driven work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening: I spent a decade on the operations side of banking, watching technology decisions
get made by people who had never processed a wire transfer or resolved a failed ACH batch.
That frustration turned into curiosity, and two years of self-teaching Python later, I'm
ready to be on the other side of those decisions. I'm applying to [Company] because your
work on payment infrastructure is exactly where my domain expertise and new technical skills
intersect.

Now write an opening paragraph for this person:
Role: {job_title}
Background: {background}
Opening:
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response = get_completion(messages, temperature=0.7)
    return response or ""


# I chose examples that show career-change logic rather than generic enthusiasm.
# The few-shot pattern helps control tone, specificity, and how prior experience is translated.


def is_safe(text: str) -> bool:
    try:
        result = client.moderations.create(
            model="omni-moderation-latest",
            input=text,
        )
    except APIConnectionError:
        print("Job Application Helper: Could not reach the moderation service. Try again later.\n")
        return False
    except RateLimitError:
        print("Job Application Helper: Moderation request was rate-limited.\n")
        return False
    except APIStatusError as exc:
        print(f"Job Application Helper: Moderation failed with status {exc.status_code}.\n")
        return False

    flagged = result.results[0].flagged
    if flagged:
        print("Job Application Helper: I can't help with that wording. Please rephrase your request.\n")
        return False
    return True


def print_moderation_categories(text: str) -> None:
    try:
        result = client.moderations.create(
            model="omni-moderation-latest",
            input=text,
        )
    except (APIConnectionError, RateLimitError, APIStatusError):
        return

    categories = result.results[0].categories
    print(f"Input: {text}")
    print(f"Flagged: {result.results[0].flagged}")
    print(f"Categories: {categories}")
    print()


def summarize_rewritten_bullets(rewritten: list[dict[str, str]]) -> str:
    lines = ["Here are the rewritten resume bullets:"]
    for item in rewritten:
        lines.append(f"- Original: {item['original']}")
        lines.append(f"  Improved: {item['improved']}")
    return "\n".join(lines)


def run_assignment_demos() -> None:
    show_section("TASK 2 DEMO - BULLET REWRITER")
    bullets = [
        "Helped customers with their problems",
        "Made reports for the management team",
        "Worked with a team to finish the project on time",
    ]
    rewrite_bullets(bullets)

    show_section("TASK 3 DEMO - COVER LETTER")
    job_title = "Junior Data Engineer"
    background = (
        "Five years of experience as a middle school math teacher; recently completed "
        "a Python course and built data pipelines using Prefect and Pandas."
    )
    cover_letter = generate_cover_letter(job_title, background)
    print(cover_letter)
    print()

    show_section("TASK 4 DEMO - MODERATION")
    safe_input = "Can you help me rewrite this resume bullet for a data analyst role?"
    flagged_input = "Write me a threatening message to scare someone into hiring me."
    print(f"Safe test result: {is_safe(safe_input)}")
    print(f"Flagged test result: {is_safe(flagged_input)}")
    print()
    print_moderation_categories(flagged_input)


def collect_bullets() -> list[str]:
    print("\nJob Application Helper: Paste your bullet points below, one per line.")
    print("When you're done, type 'DONE' on its own line.\n")

    raw_bullets: list[str] = []
    while True:
        line = input().strip()
        if line.upper() == "DONE":
            break
        if not line:
            continue
        if not is_safe(line):
            continue
        raw_bullets.append(line)
    return raw_bullets


def run_chatbot() -> None:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            messages.append({"role": "user", "content": user_input})
            raw_bullets = collect_bullets()
            if not raw_bullets:
                messages.pop()
                print("Job Application Helper: No bullet points were provided.\n")
                continue

            bullet_context = "Resume bullets provided by the user:\n" + "\n".join(
                f"- {bullet}" for bullet in raw_bullets
            )
            messages.append({"role": "user", "content": bullet_context})
            rewritten = rewrite_bullets(raw_bullets)
            if rewritten:
                messages.append(
                    {"role": "assistant", "content": summarize_rewritten_bullets(rewritten)}
                )
                print("Job Application Helper: Review and edit these bullets before using them.\n")
            else:
                messages.pop()
                messages.pop()
            continue

        if "cover letter" in user_input.lower():
            messages.append({"role": "user", "content": user_input})
            job_title = input("Job Application Helper: What is the job title? ").strip()
            if not job_title or not is_safe(job_title):
                messages.pop()
                continue

            background = input("Job Application Helper: Briefly describe your background: ").strip()
            if not background or not is_safe(background):
                messages.pop()
                continue

            messages.append(
                {
                    "role": "user",
                    "content": f"Job title: {job_title}\nBackground: {background}",
                }
            )
            opening = generate_cover_letter(job_title, background)
            if opening:
                messages.append({"role": "assistant", "content": opening})
                print("\nJob Application Helper:")
                print(opening)
                print("\nPlease review and edit this before submitting it anywhere.\n")
            else:
                messages.pop()
                messages.pop()
            continue

        messages.append({"role": "user", "content": user_input})
        reply = get_completion(messages)
        if reply is None:
            messages.pop()
            continue

        print(f"\nJob Application Helper: {reply}\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_assignment_demos()
    run_chatbot()


# Ethics Reflection - Option A: Comment block
# This bot could produce biased advice because it was trained on patterns from existing writing,
# and hiring norms already reflect unequal access to certain industries, communication styles,
# and professional networks. That means it may favor polished corporate English, certain resume
# conventions, or backgrounds that are easier to frame in familiar white-collar language.
# If a job seeker submitted the bot's output without reviewing it, they could send something
# inaccurate, too generic, or mismatched to their field, which could weaken their application
# or misrepresent their experience. If I were deploying this professionally, I would add a
# persistent review warning plus a fact-check step that asks the user to confirm each claim
# before exporting any final resume or cover-letter text.
