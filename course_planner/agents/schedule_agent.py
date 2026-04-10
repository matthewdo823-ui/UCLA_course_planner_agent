"""Schedule agent: waits for BOTH bruinwalk and grade_dist results, merges
them, generates valid schedule candidates, scores them, and forwards the
top 20 to the ranking agent.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from datetime import datetime, timezone
from uuid import uuid4

from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

from course_planner.utils import (
    AGENT_ADDRESSES,
    CourseOption,
    DaySchedule,
    ScheduleCandidate,
    Section,
    StudentProfile,
    deserialize,
    serialize,
)

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 20

# ---------------------------------------------------------------------------
# Merge-wait state
# ---------------------------------------------------------------------------

pending: dict[str, dict] = {}


def _session_key(profile: StudentProfile) -> str:
    return f"{profile.name}::{profile.major}"


def _merge_courses(
    bw_courses: list[CourseOption],
    gd_courses: list[CourseOption],
) -> list[CourseOption]:
    """Copy grade_distribution from grade_dist onto the bruinwalk list."""
    gd_map: dict[str, object] = {}
    for c in gd_courses:
        code = " ".join(c.course_code.upper().split())
        if c.grade_distribution is not None:
            gd_map[code] = c.grade_distribution

    for course in bw_courses:
        code = " ".join(course.course_code.upper().split())
        if code in gd_map:
            course.grade_distribution = gd_map[code]

    return bw_courses


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

_DAY_CHARS = {"M": "Monday", "T": "Tuesday", "W": "Wednesday",
              "R": "Thursday", "F": "Friday", "S": "Saturday", "U": "Sunday"}


def _parse_minutes(t: str) -> int | None:
    """Parse '10:00am', '2:30pm', '14:00', '10:00' -> minutes from midnight."""
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


def _section_time_blocks(sec: Section) -> list[tuple[str, int, int]]:
    """Return list of (day_name, start_min, end_min) for a section."""
    start = _parse_minutes(sec.start_time)
    end = _parse_minutes(sec.end_time)
    if start is None or end is None:
        return []
    blocks: list[tuple[str, int, int]] = []
    for ch in sec.days.upper():
        day = _DAY_CHARS.get(ch)
        if day:
            blocks.append((day, start, end))
    return blocks


def _blocks_conflict(
    a: list[tuple[str, int, int]],
    b: list[tuple[str, int, int]],
) -> bool:
    for d1, s1, e1 in a:
        for d2, s2, e2 in b:
            if d1 == d2 and s1 < e2 and s2 < e1:
                return True
    return False


# ---------------------------------------------------------------------------
# Constraint parsing from StudentProfile.hard_constraints
# ---------------------------------------------------------------------------

def _parse_constraints(profile: StudentProfile) -> dict:
    """Extract structured constraints from the free-text list."""
    days_off: set[str] = set()
    no_before: int | None = None
    no_after: int | None = None
    max_gap: int | None = None
    max_consec: int | None = None

    for c in (profile.hard_constraints or []):
        low = c.lower().strip()

        # Days off
        for day_name in ("monday", "tuesday", "wednesday", "thursday",
                         "friday", "saturday", "sunday"):
            if day_name in low and ("off" in low or "no" in low or "free" in low):
                days_off.add(day_name.capitalize())

        # No classes before X
        m = re.search(r"(?:before|earlier than)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", low)
        if m:
            mins = _parse_minutes(m.group(1))
            if mins is not None:
                no_before = mins

        # No classes after X
        m = re.search(r"(?:after|later than)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", low)
        if m:
            mins = _parse_minutes(m.group(1))
            if mins is not None:
                no_after = mins

        # Max gap
        m = re.search(r"(\d+)\s*(?:min|minute).*gap", low)
        if m:
            max_gap = int(m.group(1))

        # Max consecutive
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


# ---------------------------------------------------------------------------
# Section grouping: lecture + discussion/lab pairing
# ---------------------------------------------------------------------------

def _group_sections(course: CourseOption) -> list[list[Section]]:
    """Group sections into valid enrolment choices.

    Returns a list of options; each option is a list of 1-2 Sections
    (lecture, and optionally its discussion/lab).
    If there are only lectures, each is its own option.
    If there are discussions/labs, pair each with every lecture
    (UCLA typically lets you pick any discussion under any lecture,
    but some courses restrict — we pair all combos and rely on
    conflict filtering to prune).
    """
    lectures: list[Section] = []
    discussions: list[Section] = []

    for s in course.sections:
        st = s.section_type.lower()
        if st in ("discussion", "dis", "lab"):
            discussions.append(s)
        else:
            lectures.append(s)

    if not lectures:
        # Everything is a discussion/lab somehow — treat each as standalone
        return [[s] for s in course.sections] if course.sections else []

    if not discussions:
        return [[lec] for lec in lectures]

    # Pair each lecture with each discussion/lab
    options: list[list[Section]] = []
    for lec in lectures:
        for dis in discussions:
            # Skip if the lecture and its own discussion conflict
            lb = _section_time_blocks(lec)
            db = _section_time_blocks(dis)
            if not _blocks_conflict(lb, db):
                options.append([lec, dis])

    # Fallback: if all pairs conflict, just offer lectures
    return options if options else [[lec] for lec in lectures]


# ---------------------------------------------------------------------------
# Schedule generation
# ---------------------------------------------------------------------------


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _generate_schedules(
    courses: list[CourseOption],
    profile: StudentProfile,
) -> list[ScheduleCandidate]:
    """Build, filter, score, and return top schedule candidates."""

    constraints = _parse_constraints(profile)
    days_off = constraints["days_off"]
    no_before = constraints["no_before"]
    no_after = constraints["no_after"]
    max_gap = constraints["max_gap"]
    max_consec = constraints["max_consec"]

    preferred_fmts: set[str] = set()
    pf = (profile.format_preference or "").lower().strip()
    if pf and pf not in ("any", ""):
        preferred_fmts = {pf}

    # Separate required vs optional
    required = [c for c in courses if c.is_required]
    optional = [c for c in courses if not c.is_required]

    required_units = sum(c.units for c in required)

    # Group sections for each course
    required_options: list[list[list[Section]]] = [_group_sections(c) for c in required]
    optional_with_options = [(c, _group_sections(c)) for c in optional]

    # Skip courses with no valid section options
    required_options = [opts for opts in required_options if opts]
    optional_with_options = [(c, opts) for c, opts in optional_with_options if opts]

    min_u = profile.min_units
    max_u = profile.max_units

    # Generate course combos: required + subsets of optional
    # that hit target unit range.  Cap optional subset size to keep
    # combinatorics manageable.
    max_optional_pick = min(len(optional_with_options), 6)
    course_combos: list[list[tuple[CourseOption, list[list[Section]]]]] = []

    req_items = list(zip(required, required_options))

    for r in range(0, max_optional_pick + 1):
        for opt_subset in itertools.combinations(optional_with_options, r):
            total_u = required_units + sum(c.units for c, _ in opt_subset)
            if total_u < min_u or total_u > max_u:
                continue
            combo = list(req_items) + list(opt_subset)
            course_combos.append(combo)
            if len(course_combos) > 500:
                break
        if len(course_combos) > 500:
            break

    candidates: list[ScheduleCandidate] = []

    for combo in course_combos:
        # For each course in the combo, pick one section-group option.
        # Build cartesian product of section-group choices.
        per_course_choices: list[list[tuple[CourseOption, list[Section]]]] = []
        for course, sec_groups in combo:
            per_course_choices.append([(course, sg) for sg in sec_groups])

        # Cap the cartesian product
        n_combos = 1
        for pcc in per_course_choices:
            n_combos *= len(pcc)
            if n_combos > 2000:
                break

        if n_combos > 2000:
            # Trim each to at most 4 options
            per_course_choices = [pcc[:4] for pcc in per_course_choices]

        if not per_course_choices:
            continue

        for section_pick in itertools.product(*per_course_choices):
            # section_pick is a tuple of (CourseOption, list[Section])
            # Collect all time blocks + validate
            all_blocks: list[tuple[str, int, int]] = []
            valid = True
            course_infos: list[dict] = []

            for course, secs in section_pick:
                lec_id = ""
                dis_id = ""
                for s in secs:
                    # Format filter
                    if preferred_fmts and s.format.lower() not in preferred_fmts:
                        valid = False
                        break

                    blocks = _section_time_blocks(s)
                    if not blocks:
                        continue

                    # Check conflicts with already-placed blocks
                    if _blocks_conflict(blocks, all_blocks):
                        valid = False
                        break

                    # Check day-off / time constraints
                    for day, start, end in blocks:
                        if day in days_off:
                            valid = False
                            break
                        if no_before is not None and start < no_before:
                            valid = False
                            break
                        if no_after is not None and end > no_after:
                            valid = False
                            break
                    if not valid:
                        break

                    all_blocks.extend(blocks)

                    if s.section_type.lower() in ("discussion", "dis", "lab"):
                        dis_id = s.section_id
                    else:
                        lec_id = s.section_id

                if not valid:
                    break

                course_infos.append({
                    "course_code": course.course_code,
                    "title": course.title,
                    "units": course.units,
                    "lecture_section_id": lec_id,
                    "discussion_section_id": dis_id,
                })

            if not valid:
                continue

            # Build DaySchedule objects
            day_blocks: dict[str, list[tuple[int, int, str, str]]] = {}
            for course, secs in section_pick:
                for s in secs:
                    for day, start, end in _section_time_blocks(s):
                        day_blocks.setdefault(day, []).append(
                            (start, end, course.course_code, s.section_id)
                        )

            day_schedules: list[DaySchedule] = []
            total_gap = 0
            max_consec_any = 0

            for day, blks in sorted(day_blocks.items()):
                blks.sort()
                sec_dicts = [
                    {"course_code": cc, "section_id": sid,
                     "start_min": s, "end_min": e,
                     "instructor": "", "location": ""}
                    for s, e, cc, sid in blks
                ]

                total_min = sum(e - s for s, e, _, _ in blks)
                gap = 0
                for i in range(1, len(blks)):
                    g = blks[i][0] - blks[i - 1][1]
                    if g > 0:
                        gap += g

                # Check max_gap constraint
                if max_gap is not None:
                    for i in range(1, len(blks)):
                        if blks[i][0] - blks[i - 1][1] > max_gap:
                            valid = False
                            break

                # Compute max consecutive stretch
                consec = 0
                if blks:
                    streak_start = blks[0][0]
                    streak_end = blks[0][1]
                    for i in range(1, len(blks)):
                        if blks[i][0] <= streak_end:
                            streak_end = max(streak_end, blks[i][1])
                        else:
                            consec = max(consec, streak_end - streak_start)
                            streak_start = blks[i][0]
                            streak_end = blks[i][1]
                    consec = max(consec, streak_end - streak_start)

                if max_consec is not None and consec > max_consec:
                    valid = False

                max_consec_any = max(max_consec_any, consec)
                total_gap += gap

                day_schedules.append(DaySchedule(
                    day=day,
                    sections=sec_dicts,
                    total_minutes=total_min,
                    gap_minutes=gap,
                    max_consecutive_minutes=consec,
                ))

            if not valid:
                continue

            days_on = len(day_schedules)
            avg_gap = total_gap / days_on if days_on else 0

            # Aggregate enrichment metrics across courses in this combo
            enroll_chances: list[float] = []
            bw_scores: list[float] = []
            gpas: list[float] = []
            workloads: list[float] = []

            for course, _ in section_pick:
                ep = course.enrollment_prediction
                if ep:
                    enroll_chances.append(ep.chance_open_at_pass)
                bwc = course.bruinwalk_composite_score
                if bwc is not None:
                    bw_scores.append(bwc)
                gd = course.grade_distribution
                if gd and gd.avg_gpa > 0:
                    gpas.append(gd.avg_gpa)
                cr = course.course_ratings
                if cr and cr.avg_hours_per_week is not None:
                    workloads.append(cr.avg_hours_per_week)

            total_units = sum(ci["units"] for ci in course_infos)

            # Quality score
            quality = (
                _clamp(1 - avg_gap / 180) * 0.4
                + _clamp(1 - days_on / 5) * 0.3
                + _clamp(1 - max_consec_any / 300) * 0.3
            )

            candidates.append(ScheduleCandidate(
                courses=course_infos,
                day_schedules=day_schedules,
                total_units=total_units,
                days_on_campus=days_on,
                avg_gap_minutes_per_day=round(avg_gap, 1),
                max_consecutive_minutes_any_day=max_consec_any,
                avg_enrollment_chance=(
                    round(sum(enroll_chances) / len(enroll_chances), 4)
                    if enroll_chances else 0.0
                ),
                min_enrollment_chance=(
                    round(min(enroll_chances), 4) if enroll_chances else 1.0
                ),
                avg_bruinwalk_composite=(
                    round(sum(bw_scores) / len(bw_scores), 4)
                    if bw_scores else None
                ),
                avg_gpa=(
                    round(sum(gpas) / len(gpas), 3) if gpas else None
                ),
                min_gpa=round(min(gpas), 3) if gpas else None,
                avg_workload_hours_per_week=(
                    round(sum(workloads) / len(workloads), 1)
                    if workloads else None
                ),
                schedule_quality_score=round(quality, 4),
            ))

            if len(candidates) >= MAX_CANDIDATES * 10:
                break

        if len(candidates) >= MAX_CANDIDATES * 10:
            break

    # Sort by quality and take top N
    candidates.sort(key=lambda c: c.schedule_quality_score, reverse=True)
    return candidates[:MAX_CANDIDATES]


# ---------------------------------------------------------------------------
# Agent + Protocol
# ---------------------------------------------------------------------------

agent = Agent(
    name="schedule-agent",
    seed="schedule-agent-seed-phrase-change-me",
    port=8006,
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
    for item in msg.content:xs
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

    key = _session_key(profile)

    # --- Determine source ---
    has_bruinwalk = any(c.bruinwalk_composite_score is not None for c in courses)
    has_grade = any(c.grade_distribution is not None for c in courses)

    is_bruinwalk_sender = (sender == AGENT_ADDRESSES.get("bruinwalk"))
    is_grade_sender = (sender == AGENT_ADDRESSES.get("grade_dist"))

    source = "unknown"
    if has_bruinwalk or is_bruinwalk_sender:
        source = "bruinwalk"
    elif has_grade or is_grade_sender:
        source = "grade_dist"

    ctx.logger.info(
        "Received %d courses from %s for session %s",
        len(courses), source, key,
    )

    # --- Merge-wait logic ---
    if key not in pending:
        pending[key] = {
            "bruinwalk_courses": courses if source == "bruinwalk" else None,
            "grade_dist_courses": courses if source == "grade_dist" else None,
            "profile": profile,
        }
        ctx.logger.info("  Stashed %s data — waiting for the other source", source)
        return

    entry = pending.pop(key)

    if source == "bruinwalk":
        entry["bruinwalk_courses"] = courses
    else:
        entry["grade_dist_courses"] = courses

    bw = entry["bruinwalk_courses"] or []
    gd = entry["grade_dist_courses"] or []
    merged = _merge_courses(bw, gd) if bw else gd

    ctx.logger.info(
        "Both sources received — merged %d courses. Generating schedules …",
        len(merged),
    )

    # --- Generate schedules ---
    candidates = _generate_schedules(merged, profile)
    ctx.logger.info("Generated %d schedule candidates", len(candidates))

    # --- Serialize and forward ---
    candidates_json = serialize(candidates)
    profile_json_out = serialize(profile)

    await ctx.send(
        AGENT_ADDRESSES["ranking"],
        ChatMessage(
            content=[
                TextContent(type="text", text=candidates_json),
                TextContent(type="text", text=profile_json_out),
            ],
            msg_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
        ),
    )

    ctx.logger.info("Forwarded %d candidates to ranking agent", len(candidates))


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


agent.include(protocol, publish_manifest=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent.run()
