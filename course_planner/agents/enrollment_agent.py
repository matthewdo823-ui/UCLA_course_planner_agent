"""Enrollment agent: enriches CourseOptions with historical enrollment data
and open-seat probability predictions, then fans out to bruinwalk + grade_dist
agents in parallel.
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

from course_planner.scrapers.soc_scraper import scrape_historical_enrollment
from course_planner.utils import (
    AGENT_ADDRESSES,
    CourseOption,
    EnrollmentPrediction,
    StudentProfile,
    deserialize,
    serialize,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ASI:One client
# ---------------------------------------------------------------------------

ASI1_API_KEY = os.environ.get("ASI1_API_KEY", "")
asi_client = OpenAI(base_url="https://api.asi1.ai/v1", api_key=ASI1_API_KEY)

IMPACTED_SYSTEM = (
    "You are a UCLA enrollment expert. Given a course code, title, department, "
    "and the student's major, determine whether this is an impacted (high-demand) "
    "course for that major. Reply with ONLY a JSON object: {\"impacted\": true} "
    "or {\"impacted\": false}."
)

NOTES_SYSTEM = (
    "You are a UCLA enrollment advisor. Given historical enrollment statistics "
    "for a course, write ONE concise sentence summarizing the enrollment risk "
    "for a student trying to enroll. Be specific about fill rates and waitlist "
    "likelihood."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_impacted(course_code: str, title: str, major: str) -> bool:
    """Ask ASI:One whether this course is impacted for the student's major."""
    dept = course_code.rsplit(" ", 1)[0] if " " in course_code else course_code
    prompt = (
        f"Course: {course_code} — {title}\n"
        f"Department: {dept}\n"
        f"Student major: {major}"
    )
    try:
        resp = asi_client.chat.completions.create(
            model="asi1-mini",
            messages=[
                {"role": "system", "content": IMPACTED_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=64,
        )
        raw = (resp.choices[0].message.content or "").strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
            return bool(data.get("impacted", False))
    except Exception:
        logger.exception("Impacted-major check failed for %s", course_code)
    return False


def _generate_notes(course_code: str, prediction: EnrollmentPrediction) -> str:
    """Ask ASI:One for a one-sentence enrollment risk summary."""
    prompt = (
        f"Course: {course_code}\n"
        f"Historical quarters sampled: {prediction.historical_quarters_sampled}\n"
        f"Avg fill rate by day 1: {prediction.avg_fill_rate_by_day_1:.0%}\n"
        f"Avg fill rate by day 7: {prediction.avg_fill_rate_by_day_7:.0%}\n"
        f"Has historically gone to waitlist: {prediction.has_historically_gone_to_waitlist}\n"
        f"Current fill rate: {prediction.current_fill_rate:.0%}\n"
        f"Current waitlist count: {prediction.current_waitlist_count}\n"
        f"Class size: {prediction.class_size}\n"
        f"Impacted major course: {prediction.is_impacted_major_course}"
    )
    try:
        resp = asi_client.chat.completions.create(
            model="asi1-mini",
            messages=[
                {"role": "system", "content": NOTES_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=128,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("Notes generation failed for %s", course_code)
    return ""


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _build_prediction(
    course: CourseOption, profile: StudentProfile, historical: list[dict]
) -> EnrollmentPrediction:
    """Build an EnrollmentPrediction from historical + live data."""

    n = len(historical)

    # Averages from historical data
    avg_day1 = 0.0
    avg_day7 = 0.0
    waitlist_count = 0

    for h in historical:
        cap = h.get("capacity", 1) or 1
        avg_day1 += h.get("enrollment_day_1", 0) / cap
        avg_day7 += h.get("enrollment_day_7", 0) / cap
        if h.get("went_to_waitlist", False):
            waitlist_count += 1

    if n > 0:
        avg_day1 /= n
        avg_day7 /= n

    has_waitlisted = waitlist_count > (n / 2) if n > 0 else False

    # Average quarters until waitlist clears — heuristic from historical data
    avg_wl_clear: float | None = None
    if has_waitlisted and n > 0:
        # Rough estimate: assume waitlist clears in ~1-3 quarters on average
        avg_wl_clear = max(1.0, waitlist_count / max(n, 1) * 2.0)

    # Current live data from first section
    current_fill = 0.0
    current_wl = 0
    class_size = 0
    if course.sections:
        sec = course.sections[0]
        class_size = sec.capacity
        if sec.capacity > 0:
            current_fill = sec.enrolled / sec.capacity
        current_wl = sec.waitlist

    # Impacted major check
    is_impacted = _check_impacted(course.course_code, course.title, profile.major)

    # Compute chance_open probabilities
    base = _clamp(1.0 - avg_day1) if n > 0 else 0.8

    impacted_penalty = 0.15 if is_impacted else 0.0

    chance_pass_1 = _clamp(base - impacted_penalty)
    chance_pass_2 = _clamp(base * 0.7 - impacted_penalty)
    chance_open = _clamp(base * 0.45 - impacted_penalty)

    # chance_open_at_pass depends on the student's actual pass
    if profile.enrollment_pass == profile.enrollment_pass.PASS_1:
        chance_at_pass = chance_pass_1
    elif profile.enrollment_pass == profile.enrollment_pass.PASS_2:
        chance_at_pass = chance_pass_2
    else:
        chance_at_pass = chance_open

    pred = EnrollmentPrediction(
        historical_quarters_sampled=n,
        avg_fill_rate_by_day_1=avg_day1,
        avg_fill_rate_by_day_7=avg_day7,
        has_historically_gone_to_waitlist=has_waitlisted,
        avg_quarters_until_waitlist_clears=avg_wl_clear,
        current_fill_rate=current_fill,
        current_waitlist_count=current_wl,
        class_size=class_size,
        is_impacted_major_course=is_impacted,
        chance_open_at_pass=chance_at_pass,
        chance_open_pass_1=chance_pass_1,
        chance_open_pass_2=chance_pass_2,
        chance_open_enrollment=chance_open,
    )

    # Generate notes
    pred.notes = _generate_notes(course.course_code, pred)

    return pred


# ---------------------------------------------------------------------------
# Agent + Protocol
# ---------------------------------------------------------------------------

agent = Agent(
    name="enrollment-agent",
    seed="enrollment-agent-seed-phrase-change-me",
    port=8003,
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

    # --- Extract the two TextContent blocks ---
    text_blocks: list[str] = []
    for item in msg.content:
        if isinstance(item, TextContent):
            text_blocks.append(item.text)

    if len(text_blocks) < 2:
        ctx.logger.error(
            "Expected 2 TextContent blocks (courses + profile), got %d",
            len(text_blocks),
        )
        return

    courses_json = text_blocks[0]
    profile_json = text_blocks[1]

    # --- Deserialize ---
    try:
        courses: list[CourseOption] = deserialize(courses_json, list[CourseOption])
    except Exception:
        # dacite doesn't handle list[X] directly; parse manually
        try:
            raw_list = json.loads(courses_json)
            courses = [deserialize(json.dumps(c), CourseOption) for c in raw_list]
        except Exception:
            ctx.logger.exception("Failed to deserialize CourseOption list")
            return

    try:
        profile = deserialize(profile_json, StudentProfile)
    except Exception:
        ctx.logger.exception("Failed to deserialize StudentProfile")
        return

    ctx.logger.info(
        "Received %d courses for %s — enriching with enrollment data",
        len(courses),
        profile.name,
    )

    # --- Enrich each course with enrollment prediction ---
    for course in courses:
        ctx.logger.info("  Scraping historical enrollment for %s", course.course_code)
        historical = scrape_historical_enrollment(course.course_code)
        prediction = _build_prediction(course, profile, historical)
        course.enrollment_prediction = prediction
        ctx.logger.info(
            "    → %d quarters sampled, chance at pass: %.0f%%",
            prediction.historical_quarters_sampled,
            prediction.chance_open_at_pass * 100,
        )

    # --- Serialize enriched data ---
    enriched_courses_json = serialize(courses)
    profile_json_out = serialize(profile)

    # --- Forward to BOTH bruinwalk AND grade_dist agents simultaneously ---
    outgoing = ChatMessage(
        content=[
            TextContent(type="text", text=enriched_courses_json),
            TextContent(type="text", text=profile_json_out),
        ],
        msg_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
    )

    await ctx.send(AGENT_ADDRESSES["bruinwalk"], outgoing)

    # Send a separate ChatMessage to grade_dist (each send needs its own msg_id)
    outgoing_gd = ChatMessage(
        content=[
            TextContent(type="text", text=enriched_courses_json),
            TextContent(type="text", text=profile_json_out),
        ],
        msg_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
    )

    await ctx.send(AGENT_ADDRESSES["grade_dist"], outgoing_gd)

    ctx.logger.info(
        "Forwarded %d enriched CourseOptions to bruinwalk + grade_dist agents",
        len(courses),
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
