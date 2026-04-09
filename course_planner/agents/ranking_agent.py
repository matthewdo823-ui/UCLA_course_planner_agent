"""Ranking agent: scores, validates, and ranks ScheduleCandidates, then
forwards the top 3 to the report agent.
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
    AGENT_ADDRESSES,
    ScheduleCandidate,
    StudentProfile,
    deserialize,
    serialize,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ASI:One client
# ---------------------------------------------------------------------------

ASI1_API_KEY = os.environ.get("ASI1_API_KEY", "")
asi_client = OpenAI(base_url="https://api.asi1.ai/v1", api_key="sk_f0ac409c927e4993920e03d9ec9575e61ff6a53e201e4f1eb8fd6c69f78d7e5a")

REASON_SYSTEM = (
    "You are a UCLA enrollment advisor. Given a schedule candidate's scores, "
    "write 2-3 sentences in plain English explaining why this schedule ranked "
    "where it did. Cite the actual numeric scores. Be concise and helpful."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _normalize_weights(profile: StudentProfile) -> tuple[float, float, float, float, float]:
    """Return (w_enroll, w_prof, w_gpa, w_sched, w_work), normalised to sum 1.0."""
    w = [
        profile.weight_enrollment_chance,
        profile.weight_professor_rating,
        profile.weight_avg_gpa,
        profile.weight_schedule_quality,
        profile.weight_workload,
    ]
    total = sum(w)
    if abs(total - 1.0) > 0.01:
        logger.warning(
            "Weights sum to %.4f (expected 1.0) — normalising proportionally",
            total,
        )
        if total > 0:
            w = [x / total for x in w]
        else:
            w = [0.2] * 5
    return tuple(w)  # type: ignore[return-value]


def _compute_composite(
    cand: ScheduleCandidate,
    w_enroll: float,
    w_prof: float,
    w_gpa: float,
    w_sched: float,
    w_work: float,
) -> float:
    enroll = _clamp(cand.avg_enrollment_chance)
    prof = _clamp((cand.avg_bruinwalk_composite or 0.0) / 5.0)
    gpa = _clamp((cand.avg_gpa or 0.0) / 4.0)
    sched = _clamp(cand.schedule_quality_score)
    work = _clamp(1.0 - (cand.avg_workload_hours_per_week or 0.0) / 20.0)

    raw = (
        enroll * w_enroll
        + prof * w_prof
        + gpa * w_gpa
        + sched * w_sched
        + work * w_work
    )
    return round(_clamp(raw), 4)


def _compute_preference_match(
    cand: ScheduleCandidate,
    profile: StudentProfile,
) -> float:
    score = 0.0

    course_codes = {c["course_code"].upper() for c in cand.courses}

    # +0.2 if all required courses present
    required = {c.upper() for c in (profile.required_courses or [])}
    if required and required.issubset(course_codes):
        score += 0.2

    # +0.15 per preferred course (cap at +0.3)
    preferred = {c.upper() for c in (profile.preferred_courses or [])}
    pref_count = len(preferred & course_codes)
    score += min(pref_count * 0.15, 0.3)

    # +0.1 if no days_off violated
    if not cand.violates_days_off:
        score += 0.1

    # +0.1 if all sections match preferred formats
    if not cand.violates_format_preference:
        score += 0.1

    # +0.1 if units within target range
    if profile.min_units <= cand.total_units <= profile.max_units:
        score += 0.1

    return round(_clamp(score), 4)


# ---------------------------------------------------------------------------
# Constraint re-validation
# ---------------------------------------------------------------------------

_DAY_CHARS = {"M": "Monday", "T": "Tuesday", "W": "Wednesday",
              "R": "Thursday", "F": "Friday", "S": "Saturday", "U": "Sunday"}


def _parse_minutes(t: str) -> int | None:
    t = t.strip().lower().replace(".", "")
    m = re.match(r"(\d{1,2}):(\d{2})\s*(am|pm)?", t)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    ampm = m.group(3)
    if ampm == "pm" and h != 12:
        h += 12
    elif ampm == "am" and h == 12:
        h = 0
    return h * 60 + mi


def _parse_constraints(profile: StudentProfile) -> dict:
    days_off: set[str] = set()
    no_before: int | None = None
    no_after: int | None = None
    max_gap: int | None = None
    max_consec: int | None = None

    for c in (profile.hard_constraints or []):
        low = c.lower().strip()
        for day_name in ("monday", "tuesday", "wednesday", "thursday",
                         "friday", "saturday", "sunday"):
            if day_name in low and ("off" in low or "no" in low or "free" in low):
                days_off.add(day_name.capitalize())

        m = re.search(r"(?:before|earlier than)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", low)
        if m:
            mins = _parse_minutes(m.group(1))
            if mins is not None:
                no_before = mins

        m = re.search(r"(?:after|later than)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", low)
        if m:
            mins = _parse_minutes(m.group(1))
            if mins is not None:
                no_after = mins

        m = re.search(r"(\d+)\s*(?:min|minute).*gap", low)
        if m:
            max_gap = int(m.group(1))

        m = re.search(r"(\d+)\s*(?:min|minute).*(?:consecutive|back.to.back|straight)", low)
        if m:
            max_consec = int(m.group(1))

    return {
        "days_off": days_off,
        "no_before": no_before,
        "no_after": no_after,
        "max_gap": max_gap,
        "max_consec": max_consec,
    }


def _validate_candidate(cand: ScheduleCandidate, profile: StudentProfile) -> None:
    """Set violation flags on the candidate by re-checking constraints."""
    cons = _parse_constraints(profile)
    days_off = cons["days_off"]
    no_before = cons["no_before"]
    no_after = cons["no_after"]
    max_gap = cons["max_gap"]
    max_consec = cons["max_consec"]

    preferred_fmts: set[str] = set()
    pf = (profile.format_preference or "").lower().strip()
    if pf and pf not in ("any", ""):
        preferred_fmts = {pf}

    cand.has_time_conflicts = False
    cand.violates_days_off = False
    cand.violates_time_bounds = False
    cand.violates_max_gap = False
    cand.violates_max_consecutive = False
    cand.violates_format_preference = False

    # Collect all time blocks for conflict detection
    all_blocks: list[tuple[str, int, int]] = []

    for ds in cand.day_schedules:
        day = ds.day

        # Days off
        if day in days_off:
            cand.violates_days_off = True

        for sec in ds.sections:
            s_min = sec.get("start_min", 0)
            e_min = sec.get("end_min", 0)

            # Time bounds
            if no_before is not None and s_min < no_before:
                cand.violates_time_bounds = True
            if no_after is not None and e_min > no_after:
                cand.violates_time_bounds = True

            # Conflict check
            for d2, s2, e2 in all_blocks:
                if d2 == day and s_min < e2 and s2 < e_min:
                    cand.has_time_conflicts = True
            all_blocks.append((day, s_min, e_min))

        # Max gap
        if max_gap is not None and ds.gap_minutes > max_gap:
            cand.violates_max_gap = True

        # Max consecutive
        if max_consec is not None and ds.max_consecutive_minutes > max_consec:
            cand.violates_max_consecutive = True

    # Format preference — check if any section format doesn't match
    if preferred_fmts:
        for ds in cand.day_schedules:
            for sec in ds.sections:
                fmt = sec.get("format", "in-person")
                if isinstance(fmt, str) and fmt.lower() not in preferred_fmts:
                    cand.violates_format_preference = True
                    break


# ---------------------------------------------------------------------------
# LLM reason generation
# ---------------------------------------------------------------------------

def _generate_reason(cand: ScheduleCandidate, rank: int) -> str:
    courses = ", ".join(c["course_code"] for c in cand.courses)
    prompt = (
        f"Rank #{rank} schedule with courses: {courses}\n"
        f"Composite score: {cand.composite_score:.2f}\n"
        f"Schedule quality: {cand.schedule_quality_score:.2f}\n"
        f"Avg enrollment chance: {cand.avg_enrollment_chance:.0%}\n"
        f"Avg GPA: {cand.avg_gpa or 'N/A'}\n"
        f"Avg Bruinwalk composite: {cand.avg_bruinwalk_composite or 'N/A'}\n"
        f"Avg workload: {cand.avg_workload_hours_per_week or 'N/A'} hrs/week\n"
        f"Days on campus: {cand.days_on_campus}\n"
        f"Total units: {cand.total_units}\n"
        f"Preference match: {cand.preference_match_score:.2f}"
    )
    try:
        resp = asi_client.chat.completions.create(
            model="asi1-mini",
            messages=[
                {"role": "system", "content": REASON_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("Reason generation failed for rank %d", rank)
    return ""


# ---------------------------------------------------------------------------
# Agent + Protocol
# ---------------------------------------------------------------------------

agent = Agent(
    name="ranking-agent",
    seed="ranking-agent-seed-phrase-change-me",
    port=8007,
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
            "Expected 2 TextContent blocks (candidates + profile), got %d",
            len(text_blocks),
        )
        return

    candidates_json = text_blocks[0]
    profile_json = text_blocks[1]

    # --- Deserialize ---
    try:
        raw_list = json.loads(candidates_json)
        candidates: list[ScheduleCandidate] = [
            deserialize(json.dumps(c), ScheduleCandidate) for c in raw_list
        ]
    except Exception:
        ctx.logger.exception("Failed to deserialize ScheduleCandidate list")
        return

    try:
        profile = deserialize(profile_json, StudentProfile)
    except Exception:
        ctx.logger.exception("Failed to deserialize StudentProfile")
        return

    ctx.logger.info(
        "Received %d candidates for %s — ranking",
        len(candidates), profile.name,
    )

    # --- Validate & normalise weights ---
    w_enroll, w_prof, w_gpa, w_sched, w_work = _normalize_weights(profile)
    ctx.logger.info(
        "Weights: enroll=%.2f prof=%.2f gpa=%.2f sched=%.2f work=%.2f",
        w_enroll, w_prof, w_gpa, w_sched, w_work,
    )

    # --- Score, validate, and rank ---
    for cand in candidates:
        cand.composite_score = _compute_composite(
            cand, w_enroll, w_prof, w_gpa, w_sched, w_work
        )
        _validate_candidate(cand, profile)
        cand.preference_match_score = _compute_preference_match(cand, profile)

    # Sort descending by composite_score
    candidates.sort(key=lambda c: c.composite_score, reverse=True)

    # Assign ranks
    for i, cand in enumerate(candidates, start=1):
        cand.rank = i

    ctx.logger.info(
        "Top 3 scores: %s",
        [c.composite_score for c in candidates[:3]],
    )

    # --- Generate reasons for top 3 ---
    top3 = candidates[:3]
    for cand in top3:
        ctx.logger.info("  Generating reason for rank #%d", cand.rank)
        cand.reason = _generate_reason(cand, cand.rank)

    # --- Serialize and forward top 3 to report agent ---
    top3_json = serialize(top3)
    profile_json_out = serialize(profile)

    await ctx.send(
        AGENT_ADDRESSES["report"],
        ChatMessage(
            content=[
                TextContent(type="text", text=top3_json),
                TextContent(type="text", text=profile_json_out),
            ],
            msg_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
        ),
    )

    ctx.logger.info("Forwarded top 3 ranked candidates to report agent")


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


agent.include(protocol, publish_manifest=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent.run()
