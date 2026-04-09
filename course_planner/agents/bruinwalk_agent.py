"""Bruinwalk agent: enriches CourseOptions with professor and course ratings
from Bruinwalk, computes a composite score, then forwards to the schedule agent.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

from course_planner.scrapers.bruinwalk_scraper import (
    scrape_course_ratings,
    scrape_professor_ratings,
)
from course_planner.utils import (
    AGENT_ADDRESSES,
    CourseOption,
    StudentProfile,
    deserialize,
    serialize,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_composite(course: CourseOption) -> float | None:
    """Composite = avg(professor overall_ratings) * 0.6 + course_rating * 0.4.

    If only one source is available, use it alone. If neither, return None.
    """
    prof_avg: float | None = None
    if course.professor_ratings:
        vals = [
            r.overall_rating
            for r in course.professor_ratings.values()
            if r.overall_rating is not None
        ]
        if vals:
            prof_avg = sum(vals) / len(vals)

    course_val: float | None = None
    if course.course_ratings and course.course_ratings.overall_course_rating is not None:
        course_val = course.course_ratings.overall_course_rating

    if prof_avg is not None and course_val is not None:
        return round(prof_avg * 0.6 + course_val * 0.4, 4)
    if prof_avg is not None:
        return round(prof_avg, 4)
    if course_val is not None:
        return round(course_val, 4)
    return None


# ---------------------------------------------------------------------------
# Agent + Protocol
# ---------------------------------------------------------------------------

agent = Agent(
    name="bruinwalk-agent",
    seed="bruinwalk-agent-seed-phrase-change-me",
    port=8004,
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

    # --- Extract two TextContent blocks ---
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
        raw_list = json.loads(courses_json)
        courses: list[CourseOption] = [
            deserialize(json.dumps(c), CourseOption) for c in raw_list
        ]
    except Exception:
        ctx.logger.exception("Failed to deserialize CourseOption list")
        return

    try:
        profile = deserialize(profile_json, StudentProfile)
    except Exception:
        ctx.logger.exception("Failed to deserialize StudentProfile")
        return

    ctx.logger.info(
        "Received %d courses for %s — enriching with Bruinwalk ratings",
        len(courses),
        profile.name,
    )

    # --- Enrich each course ---
    for course in courses:
        # Scrape course ratings (once per course)
        ctx.logger.info("  Scraping course ratings for %s", course.course_code)
        cr = scrape_course_ratings(course.course_code)
        course.course_ratings = cr

        # Scrape professor ratings for each unique instructor
        prof_ratings: dict[str, object] = {}
        seen_instructors: set[str] = set()
        for section in course.sections:
            name = section.instructor.strip()
            if not name or name in seen_instructors:
                continue
            seen_instructors.add(name)
            ctx.logger.info("    Scraping professor ratings for %s", name)
            pr = scrape_professor_ratings(name, course.course_code)
            if pr is not None:
                prof_ratings[name] = pr

        course.professor_ratings = prof_ratings if prof_ratings else None

        # Compute composite score
        course.bruinwalk_composite_score = _compute_composite(course)

        ctx.logger.info(
            "    → composite score: %s",
            course.bruinwalk_composite_score,
        )

    # --- Serialize and forward to schedule agent ---
    enriched_courses_json = serialize(courses)
    profile_json_out = serialize(profile)

    await ctx.send(
        AGENT_ADDRESSES["schedule"],
        ChatMessage(
            content=[
                TextContent(type="text", text=enriched_courses_json),
                TextContent(type="text", text=profile_json_out),
            ],
            msg_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
        ),
    )

    ctx.logger.info(
        "Forwarded %d enriched CourseOptions to schedule agent",
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
