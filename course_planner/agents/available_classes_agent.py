"""Available-classes agent: receives a StudentProfile, scrapes the UCLA SOC,
filters courses by eligibility/prerequisites, and forwards CourseOptions to
the enrollment agent.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from uuid import uuid4

from openai import OpenAI
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

from course_planner.scrapers.soc_scraper import scrape_quarter_courses
from course_planner.utils import (
    AGENT_ADDRESSES,
    CourseOption,
    Section,
    StudentProfile,
    deserialize,
    serialize,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ASI:One client (for prerequisite extraction)
# ---------------------------------------------------------------------------

ASI1_API_KEY = os.environ.get("ASI1_API_KEY", "")
asi_client = OpenAI(base_url="https://api.asi1.ai/v1", api_key="sk_f0ac409c927e4993920e03d9ec9575e61ff6a53e201e4f1eb8fd6c69f78d7e5a")

PREREQ_SYSTEM = (
    "Given this UCLA course description, extract the list of prerequisite "
    "course codes as a JSON array. Return [] if none. Description: "
)


def _extract_prerequisites(description: str) -> list[str]:
    """Use ASI:One to pull prerequisite course codes from a description."""
    if not description.strip():
        return []
    try:
        resp = asi_client.chat.completions.create(
            model="asi1-mini",
            messages=[
                {"role": "system", "content": PREREQ_SYSTEM},
                {"role": "user", "content": description},
            ],
            max_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        logger.exception("Prerequisite extraction failed")
    return []


# ---------------------------------------------------------------------------
# Core filtering logic
# ---------------------------------------------------------------------------


def _normalize_code(code: str) -> str:
    """Normalize a course code for comparison (strip spaces, upper)."""
    return " ".join(code.upper().split())


def _filter_courses(
    raw_courses: list[dict],
    profile: StudentProfile,
) -> list[CourseOption]:
    """Apply eligibility filters and return CourseOption instances."""
    completed = {_normalize_code(c) for c in (profile.dars_courses or [])}
    required = {_normalize_code(c) for c in (profile.required_courses or [])}
    preferred = {_normalize_code(c) for c in (profile.preferred_courses or [])}

    # Determine acceptable formats
    pref_fmt = (profile.format_preference or "").lower().strip()
    acceptable_formats: set[str] = set()
    if pref_fmt and pref_fmt != "any":
        acceptable_formats = {pref_fmt}

    results: list[CourseOption] = []

    for raw in raw_courses:
        code = _normalize_code(raw.get("course_code", ""))
        if not code:
            continue

        # Skip already-completed courses
        if code in completed:
            logger.debug("Skipping %s — already completed", code)
            continue

        # Check prerequisites
        prereqs = _extract_prerequisites(raw.get("description", ""))
        prereqs_met = all(_normalize_code(p) in completed for p in prereqs)

        # Filter sections by format preference
        raw_sections = raw.get("sections", [])
        if acceptable_formats:
            raw_sections = [
                s for s in raw_sections
                if s.get("format", "in-person").lower() in acceptable_formats
            ]
            # If all sections are filtered out, skip the course entirely
            if not raw_sections:
                continue

        # Build Section dataclass instances
        sections = [
            Section(
                section_id=s.get("section_id", ""),
                days=s.get("days", ""),
                start_time=s.get("start_time", ""),
                end_time=s.get("end_time", ""),
                location=s.get("location", ""),
                instructor=s.get("instructor", ""),
                enrolled=int(s.get("enrolled", 0)),
                capacity=int(s.get("capacity", 0)),
                waitlist=int(s.get("waitlist", 0)),
                waitlist_capacity=int(s.get("waitlist_capacity", 0)),
                format=s.get("format", "in-person"),
            )
            for s in raw_sections
        ]

        results.append(
            CourseOption(
                course_code=code,
                title=raw.get("title", ""),
                units=float(raw.get("units", 4.0)),
                description=raw.get("description", ""),
                sections=sections,
                prerequisites_met=prereqs_met,
                is_required=code in required,
                is_preferred=code in preferred,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Agent + Protocol
# ---------------------------------------------------------------------------

agent = Agent(
    name="available-classes-agent",
    seed="available-classes-agent-seed-phrase-change-me",
    port=8002,
    mailbox=True,
    publish_agent_details=True,
)

protocol = Protocol(spec=chat_protocol_spec)


@protocol.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    # --- ACK ---
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id,
        ),
    )

    # --- Extract text ---
    text = ""
    for item in msg.content:
        if isinstance(item, TextContent):
            text += item.text
    text = text.strip()
    if not text:
        return

    # --- Deserialize the incoming StudentProfile ---
    try:
        profile = deserialize(text, StudentProfile)
    except Exception:
        ctx.logger.exception("Failed to deserialize StudentProfile")
        return

    ctx.logger.info(
        "Received profile for %s (%s) — scraping SOC",
        profile.name,
        profile.major,
    )

    # --- Determine departments to scrape ---
    departments_to_scrape: set[str] = set()

    major_upper = profile.major.upper()
    if "COMPUTER SCIENCE" in major_upper or "COM SCI" in major_upper:
        departments_to_scrape.add("COM SCI")
    elif "MATH" in major_upper:
        departments_to_scrape.add("MATH")
    elif "PHYSICS" in major_upper:
        departments_to_scrape.add("PHYSICS")
    elif "ECON" in major_upper:
        departments_to_scrape.add("ECON")
    else:
        departments_to_scrape.add(major_upper)

    # Also scrape departments for explicitly required/preferred courses
    for code in (profile.required_courses or []) + (profile.preferred_courses or []):
        parts = code.strip().upper().rsplit(" ", 1)
        if len(parts) >= 2:
            departments_to_scrape.add(parts[0])

    quarter = "Fall 2025"

    # --- Scrape ---
    all_raw: list[dict] = []
    for dept in departments_to_scrape:
        ctx.logger.info("Scraping SOC: %s / %s", quarter, dept)
        raw = scrape_quarter_courses(quarter, dept)
        all_raw.extend(raw)
        ctx.logger.info("  → %d courses found", len(raw))

    # --- Filter & build CourseOptions ---
    course_options = _filter_courses(all_raw, profile)
    ctx.logger.info(
        "Filtered to %d eligible course options", len(course_options)
    )

    # --- Serialize and forward to enrollment agent ---
    # Two TextContent blocks: courses JSON + profile JSON
    courses_json = serialize(course_options)
    profile_json = serialize(profile)

    await ctx.send(
        AGENT_ADDRESSES["enrollment"],
        ChatMessage(
            content=[
                TextContent(type="text", text=courses_json),
                TextContent(type="text", text=profile_json),
            ],
            msg_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
        ),
    )

    ctx.logger.info(
        "Forwarded %d CourseOptions + StudentProfile to enrollment agent",
        len(course_options),
    )


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


agent.include(protocol, publish_manifest=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent.run()
