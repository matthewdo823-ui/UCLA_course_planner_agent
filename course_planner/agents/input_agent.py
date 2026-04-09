"""Input agent: multi-turn conversational intake that builds a StudentProfile.

This is the only agent the user talks to directly. It collects all required
fields via natural dialogue driven by ASI:One, parses the DARS report, then
forwards the completed profile to the available_classes_agent.
"""

from __future__ import annotations

import json
import logging
import os
import re
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

from course_planner.utils import (
    EnrollmentPass,
    StudentProfile,
    YearLevel,
)
from course_planner.utils import AGENT_ADDRESSES, serialize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ASI:One client
# ---------------------------------------------------------------------------

ASI1_API_KEY = os.environ.get("ASI1_API_KEY", "")
asi_client = OpenAI(base_url="https://api.asi1.ai/v1", api_key="sk_f0ac409c927e4993920e03d9ec9575e61ff6a53e201e4f1eb8fd6c69f78d7e5a")

# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

# Fields collected in order during the conversation.
FIELD_ORDER: list[str] = [
    "name_major",        # 1. student name + major
    "year_gpa",          # 2. year level + GPA
    "units_completed",   # 3. total units completed
    "enrollment_pass",   # 4. pass type + open datetime
    "dars_text",         # 5. DARS report paste
    "required_courses",  # 6. must-take courses
    "preferred_courses", # 7. nice-to-have courses
    "constraints",       # 8. scheduling constraints
    "format_pref",       # 9. in-person / hybrid / online
    "unit_range",        # 10. desired min and max units
]

# In-memory session store keyed by sender address.
sessions: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

INTAKE_SYSTEM = """\
You are a friendly UCLA course-planning assistant collecting student information.
You are having a multi-turn conversation to gather the student's details.
Always be concise, clear, and encouraging. Ask only for the specific information
indicated in the assistant instructions — do not ask for anything extra.
If the student's answer is ambiguous, ask a brief clarifying follow-up.
When you have the information you need, confirm it back to the student naturally.\
"""

DARS_EXTRACT_SYSTEM = """\
Extract all completed course codes from this UCLA DARS report.
Return ONLY a JSON array of strings in the format ['CS 111', 'MATH 33A', ...].
Include every course listed as completed or in-progress.\
"""

# Per-field prompts that tell the LLM what to ask for next.
FIELD_PROMPTS: dict[str, str] = {
    "name_major": (
        "Ask the student for their full name and their major at UCLA."
    ),
    "year_gpa": (
        "Ask the student what year they are "
        "(freshman, sophomore, junior, senior, or graduate) and their current GPA."
    ),
    "units_completed": (
        "Ask the student how many total units they have completed so far."
    ),
    "enrollment_pass": (
        "Ask the student which enrollment pass they are on "
        "(Pass 1, Pass 2, or Open Enrollment) and the exact date and time "
        "their pass opens."
    ),
    "dars_text": (
        "Ask the student to paste their DARS report as plain text — "
        "they should copy everything under 'COURSES COMPLETED' from their "
        "MyUCLA PDF and paste it here."
    ),
    "required_courses": (
        "Ask the student if there are any specific courses they MUST take next quarter. "
        "These are required courses they need to enroll in."
    ),
    "preferred_courses": (
        "Ask the student if there are any courses they would prefer to take if possible, "
        "but that are not strictly required."
    ),
    "constraints": (
        "Ask the student about any hard scheduling constraints: "
        "days they want off, the earliest time they'd want a class, "
        "and the latest time they'd want a class to end."
    ),
    "format_pref": (
        "Ask the student their preferred class format: in-person, hybrid, or online."
    ),
    "unit_range": (
        "Ask the student how many units they'd like to take — "
        "specifically, the minimum and maximum number of units they want."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_session() -> dict:
    """Return a blank session dict."""
    return {
        "step": 0,
        "history": [],  # list of {"role": ..., "content": ...}
        "collected": {},  # field_name -> raw answer text
    }


def _current_field(session: dict) -> str | None:
    step = session["step"]
    if step < len(FIELD_ORDER):
        return FIELD_ORDER[step]
    return None


async def _llm_chat(history: list[dict], system_extra: str = "") -> str:
    """Call ASI:One with the running conversation history."""
    system = INTAKE_SYSTEM
    if system_extra:
        system += "\n\n" + system_extra
    messages = [{"role": "system", "content": system}] + history
    try:
        resp = asi_client.chat.completions.create(
            model="asi1-mini",
            messages=messages,
            max_tokens=512,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.exception("ASI:One call failed")
        return f"(LLM unavailable — please try again: {exc})"


async def _parse_dars(dars_text: str) -> list[str]:
    """Use ASI:One to extract course codes from pasted DARS text."""
    try:
        resp = asi_client.chat.completions.create(
            model="asi1-mini",
            messages=[
                {"role": "system", "content": DARS_EXTRACT_SYSTEM},
                {"role": "user", "content": dars_text},
            ],
            max_tokens=2048,
        )
        raw = (resp.choices[0].message.content or "").strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        logger.exception("DARS parsing failed")
    return []


def _enrollment_pass_from_str(s: str) -> EnrollmentPass:
    low = s.lower()
    if "1" in low or "one" in low:
        return EnrollmentPass.PASS_1
    if "2" in low or "two" in low:
        return EnrollmentPass.PASS_2
    return EnrollmentPass.OPEN


def _build_profile(collected: dict, dars_courses: list[str], reply_to: str = "") -> StudentProfile:
    """Construct a StudentProfile from the collected conversation data."""
    # Parse year
    year_raw = collected.get("year_gpa", "junior").lower()
    year = YearLevel.JUNIOR
    for y in YearLevel:
        if y.value in year_raw:
            year = y
            break

    # Parse GPA
    gpa = 0.0
    gpa_match = re.search(r"(\d+\.\d+)", collected.get("year_gpa", ""))
    if gpa_match:
        gpa = float(gpa_match.group(1))

    # Parse units completed
    units = 0.0
    units_match = re.search(r"(\d+\.?\d*)", collected.get("units_completed", "0"))
    if units_match:
        units = float(units_match.group(1))

    # Parse enrollment pass
    pass_raw = collected.get("enrollment_pass", "")
    enrollment_pass = _enrollment_pass_from_str(pass_raw)

    # Parse unit range
    unit_range_raw = collected.get("unit_range", "12-16")
    unit_nums = re.findall(r"(\d+)", unit_range_raw)
    min_units = int(unit_nums[0]) if len(unit_nums) >= 1 else 12
    max_units = int(unit_nums[1]) if len(unit_nums) >= 2 else min_units

    # Parse required/preferred courses
    required = [
        c.strip()
        for c in re.split(r"[,;\n]", collected.get("required_courses", ""))
        if c.strip() and c.strip().lower() not in ("none", "n/a", "no")
    ]
    preferred = [
        c.strip()
        for c in re.split(r"[,;\n]", collected.get("preferred_courses", ""))
        if c.strip() and c.strip().lower() not in ("none", "n/a", "no")
    ]

    # Parse constraints
    constraints = [
        c.strip()
        for c in re.split(r"[,;\n]", collected.get("constraints", ""))
        if c.strip() and c.strip().lower() not in ("none", "n/a", "no")
    ]

    return StudentProfile(
        name=collected.get("name_major", ""),
        major=collected.get("name_major", "Undeclared"),
        year=year,
        gpa=gpa,
        units_completed=units,
        enrollment_pass=enrollment_pass,
        pass_open_datetime=collected.get("enrollment_pass", ""),
        dars_text=collected.get("dars_text"),
        dars_courses=dars_courses,
        required_courses=required,
        preferred_courses=preferred,
        hard_constraints=constraints,
        format_preference=collected.get("format_pref", "in-person"),
        min_units=min_units,
        max_units=max_units,
        reply_to_user=reply_to,
    )


# ---------------------------------------------------------------------------
# Agent + Protocol
# ---------------------------------------------------------------------------

agent = Agent(
    name="input-agent",
    seed="input-agent-seed-phrase-change-me",
    port=8001,
    mailbox=True,
    publish_agent_details=True,
)

protocol = Protocol(spec=chat_protocol_spec)


async def _send_text(ctx: Context, destination: str, text: str) -> None:
    """Send a ChatMessage containing a single TextContent."""
    await ctx.send(
        destination,
        ChatMessage(
            content=[TextContent(type="text", text=text)],
            msg_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
        ),
    )


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

    # --- Extract user text ---
    user_text = ""
    for item in msg.content:
        if isinstance(item, TextContent):
            user_text += item.text
    user_text = user_text.strip()
    if not user_text:
        return

    # --- Session lookup / create ---
    if sender not in sessions:
        sessions[sender] = _new_session()
    session = sessions[sender]

    current_field = _current_field(session)

    # ------------------------------------------------------------------
    # First message: generate the opening question
    # ------------------------------------------------------------------
    if session["step"] == 0 and not session["history"]:
        prompt_instruction = FIELD_PROMPTS[FIELD_ORDER[0]]
        session["history"].append({"role": "user", "content": user_text})
        opening = await _llm_chat(
            session["history"],
            system_extra=f"Your next task: {prompt_instruction}",
        )
        session["history"].append({"role": "assistant", "content": opening})
        await _send_text(ctx, sender, opening)
        return

    # ------------------------------------------------------------------
    # Continuing conversation: store the user's answer for current field
    # ------------------------------------------------------------------
    session["history"].append({"role": "user", "content": user_text})

    if current_field is not None:
        session["collected"][current_field] = user_text

        # Special handling: DARS parsing
        if current_field == "dars_text":
            dars_courses = await _parse_dars(user_text)
            session["collected"]["_dars_courses"] = dars_courses
            ctx.logger.info(f"DARS parsed — {len(dars_courses)} courses extracted")

        # Advance to next field
        session["step"] += 1

    next_field = _current_field(session)

    # ------------------------------------------------------------------
    # All fields collected → build profile and forward
    # ------------------------------------------------------------------
    if next_field is None:
        dars_courses = session["collected"].get("_dars_courses", [])
        profile = _build_profile(session["collected"], dars_courses, reply_to=sender)
        profile_json = serialize(profile)

        ctx.logger.info("StudentProfile complete — forwarding to available_classes_agent")

        # Forward to available_classes_agent
        await _send_text(
            ctx,
            AGENT_ADDRESSES["available_classes"],
            profile_json,
        )

        # Confirm to the user — session stays open (no EndSessionContent)
        confirmation = (
            "Thanks! I've got everything I need. Your profile is being processed — "
            "I'll get back to you with course recommendations shortly."
        )
        await _send_text(ctx, sender, confirmation)
        return

    # ------------------------------------------------------------------
    # Ask for the next field
    # ------------------------------------------------------------------
    prompt_instruction = FIELD_PROMPTS[next_field]
    reply = await _llm_chat(
        session["history"],
        system_extra=f"Your next task: {prompt_instruction}",
    )
    session["history"].append({"role": "assistant", "content": reply})
    await _send_text(ctx, sender, reply)


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


agent.include(protocol, publish_manifest=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent.run()
