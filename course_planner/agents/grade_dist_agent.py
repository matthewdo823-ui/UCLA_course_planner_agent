"""Grade-distribution agent: enriches CourseOptions with historical grade data
from UCLA grade-distribution sheets, then forwards to the schedule agent.
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

from course_planner.scrapers.grade_dist_scraper import load_grade_data
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


def _normalize_code(code: str) -> str:
    return " ".join(code.upper().split())


def _normalize_instructor(name: str) -> str:
    return name.strip()


# ---------------------------------------------------------------------------
# Agent + Protocol
# ---------------------------------------------------------------------------

agent = Agent(
    name="grade-dist-agent",
    seed="grade-dist-agent-seed-phrase-change-me",
    port=8005,
    mailbox=True,
    publish_agent_details=True,
)

protocol = Protocol(spec=chat_protocol_spec)

# Pre-load grade data at startup so the first request is fast.
logger.info("Pre-loading grade distribution data …")
_grade_data = load_grade_data()
logger.info("Grade data ready — %d courses loaded", len(_grade_data))


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
        "Received %d courses for %s — attaching grade distributions",
        len(courses),
        profile.name,
    )

    # --- Look up grade distributions ---
    for course in courses:
        code = _normalize_code(course.course_code)
        code_data = _grade_data.get(code)
        if not code_data:
            ctx.logger.debug("  No grade data for %s", code)
            continue

        # Try to match by instructor from the first section
        matched = False
        for section in course.sections:
            instructor = _normalize_instructor(section.instructor)
            if not instructor:
                continue
            # Try exact match first
            dist = code_data.get(instructor)
            if dist:
                course.grade_distribution = dist
                matched = True
                ctx.logger.info(
                    "  %s / %s → avg GPA %.2f (%d quarters)",
                    code, instructor,
                    dist.avg_gpa, dist.quarters_sampled,
                )
                break
            # Try last-name match
            last_name = instructor.split()[-1] if instructor else ""
            for instr_key, dist_val in code_data.items():
                if last_name and last_name.lower() in instr_key.lower():
                    course.grade_distribution = dist_val
                    matched = True
                    ctx.logger.info(
                        "  %s / %s (fuzzy → %s) → avg GPA %.2f",
                        code, instructor, instr_key, dist_val.avg_gpa,
                    )
                    break
            if matched:
                break

        # If no instructor match, use any available data for the course
        if not matched and code_data:
            any_dist = next(iter(code_data.values()))
            course.grade_distribution = any_dist
            ctx.logger.info(
                "  %s → using general data, avg GPA %.2f",
                code, any_dist.avg_gpa,
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
        "Forwarded %d CourseOptions with grade data to schedule agent",
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
