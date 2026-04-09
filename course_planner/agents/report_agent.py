"""Report agent: builds the final Markdown report with risk flags, fallback
suggestions, and methodology, then sends it back to the original user.
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
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)

from course_planner.utils import (
    AGENT_ADDRESSES,
    CourseRiskFlag,
    PlannerReport,
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


# ---------------------------------------------------------------------------
# Risk-flag generation
# ---------------------------------------------------------------------------

def _llm_explain(prompt: str) -> str:
    """One-sentence explanation via ASI:One."""
    try:
        resp = asi_client.chat.completions.create(
            model="asi1-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a UCLA enrollment advisor. Write exactly ONE concise "
                    "sentence explaining the risk. Mention the actual number and "
                    "what it means for the student."
                )},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("LLM explain failed")
    return ""


def _build_risk_flags(candidates: list[ScheduleCandidate]) -> list[CourseRiskFlag]:
    """Scan all courses in all candidates and build risk flags."""
    flags: list[CourseRiskFlag] = []
    seen: set[tuple[str, str]] = set()  # (course_code, flag_type) de-dup

    for cand in candidates:
        for c_dict in cand.courses:
            code = c_dict.get("course_code", "")

            # The ScheduleCandidate.courses are dicts — enrichment data was
            # on the CourseOption objects before they became schedule dicts.
            # We stored key metrics on the ScheduleCandidate aggregate fields,
            # but per-course detail may be in extended dict keys if present.
            # We'll check the candidate-level aggregates and per-course dicts.

            # --- Enrollment risk (from per-course dict if available) ---
            chance = c_dict.get("enrollment_chance")
            if chance is None:
                # Fall back to candidate avg
                chance = cand.avg_enrollment_chance

            if chance is not None and ("enrollment", code) not in seen:
                if chance < 0.50:
                    sev = "critical" if chance < 0.30 else "warning"
                    flags.append(CourseRiskFlag(
                        course_code=code,
                        flag_type="enrollment",
                        severity=sev,
                        metric_name="chance_open_at_pass",
                        metric_value=round(chance, 4),
                        threshold_used=0.50,
                        explanation=_llm_explain(
                            f"Course {code} has a {chance:.0%} chance of being "
                            f"open at the student's enrollment pass. "
                            f"Threshold for flagging is 50%."
                        ),
                    ))
                    seen.add(("enrollment", code))

            # --- Grade risk ---
            pct_df = c_dict.get("pct_d_or_f")
            if pct_df is not None and ("grade", code) not in seen:
                if pct_df > 0.25:
                    flags.append(CourseRiskFlag(
                        course_code=code,
                        flag_type="grade",
                        severity="critical",
                        metric_name="pct_d_or_f",
                        metric_value=round(pct_df, 4),
                        threshold_used=0.25,
                        explanation=_llm_explain(
                            f"Course {code} has {pct_df:.0%} of students "
                            f"receiving D or F grades, which is critically high."
                        ),
                    ))
                    seen.add(("grade", code))
                elif pct_df > 0.15:
                    flags.append(CourseRiskFlag(
                        course_code=code,
                        flag_type="grade",
                        severity="warning",
                        metric_name="pct_d_or_f",
                        metric_value=round(pct_df, 4),
                        threshold_used=0.15,
                        explanation=_llm_explain(
                            f"Course {code} has {pct_df:.0%} of students "
                            f"receiving D or F grades."
                        ),
                    ))
                    seen.add(("grade", code))

            # --- Workload risk ---
            workload = c_dict.get("workload_hours_per_week")
            if workload is not None and workload > 15 and ("workload", code) not in seen:
                flags.append(CourseRiskFlag(
                    course_code=code,
                    flag_type="workload",
                    severity="warning",
                    metric_name="workload_hours_per_week",
                    metric_value=round(workload, 1),
                    threshold_used=15.0,
                    explanation=_llm_explain(
                        f"Course {code} averages {workload:.1f} hours/week "
                        f"of work, exceeding the 15-hour warning threshold."
                    ),
                ))
                seen.add(("workload", code))

            # --- Rating risk ---
            bw = c_dict.get("bruinwalk_composite_score")
            if bw is not None and bw < 2.5 and ("rating", code) not in seen:
                flags.append(CourseRiskFlag(
                    course_code=code,
                    flag_type="rating",
                    severity="warning",
                    metric_name="bruinwalk_composite_score",
                    metric_value=round(bw, 4),
                    threshold_used=2.5,
                    explanation=_llm_explain(
                        f"Course {code} has a Bruinwalk composite score "
                        f"of {bw:.2f}/5.0, which is below the 2.5 threshold."
                    ),
                ))
                seen.add(("rating", code))

    # Also check candidate-level metrics for enrollment
    for cand in candidates:
        if cand.min_enrollment_chance < 0.50:
            # Already handled per-course above where possible;
            # add a general flag if not already covered
            pass

    # Sort: critical first, then by course code
    flags.sort(key=lambda f: (0 if f.severity == "critical" else 1, f.course_code))
    return flags


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def _llm_narrative(system: str, prompt: str, max_tokens: int = 300) -> str:
    """Generate a narrative section via ASI:One."""
    try:
        resp = asi_client.chat.completions.create(
            model="asi1-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("LLM narrative generation failed")
    return ""


def _format_time(minutes: int) -> str:
    """Convert minutes-from-midnight to HH:MM AM/PM."""
    h = minutes // 60
    m = minutes % 60
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {ampm}"


def _build_markdown(
    candidates: list[ScheduleCandidate],
    profile: StudentProfile,
    flags: list[CourseRiskFlag],
    now_str: str,
) -> str:
    """Build the full Markdown report string."""
    parts: list[str] = []

    # ---- Header ----
    parts.append(f"# UCLA Course Planner Report")
    parts.append(f"**Student:** {profile.name}  ")
    parts.append(f"**Major:** {profile.major}  ")
    parts.append(f"**Generated:** {now_str}  ")
    parts.append(f"**Enrollment Pass:** {profile.enrollment_pass.value}  ")
    parts.append(f"**Pass Opens:** {profile.pass_open_datetime}  ")
    parts.append("")

    # ---- Executive Summary (LLM) ----
    top = candidates[0] if candidates else None
    summary_data = ""
    if top:
        codes = ", ".join(c["course_code"] for c in top.courses)
        summary_data = (
            f"Student: {profile.name}, {profile.major} {profile.year.value}\n"
            f"Top schedule: {codes}\n"
            f"Composite score: {top.composite_score:.2f}, "
            f"Units: {top.total_units}, Days on campus: {top.days_on_campus}\n"
            f"Avg enrollment chance: {top.avg_enrollment_chance:.0%}\n"
            f"Risk flags: {len(flags)} ({sum(1 for f in flags if f.severity == 'critical')} critical)\n"
            f"Preference match: {top.preference_match_score:.2f}"
        )
    exec_summary = _llm_narrative(
        "You are writing the executive summary of a UCLA course planning report. "
        "Write 2-3 sentences in plain English summarizing the recommendation.",
        summary_data,
    )
    parts.append("## Executive Summary")
    parts.append(exec_summary)
    parts.append("")

    # ---- Recommended Schedule (#1) ----
    if top:
        parts.append("## Recommended Schedule (Rank #1)")
        parts.append(f"**Composite Score:** {top.composite_score:.2f} | "
                     f"**Total Units:** {top.total_units} | "
                     f"**Days on Campus:** {top.days_on_campus}")
        parts.append("")
        parts.append("| Course | Title | Units | Enrollment Chance | "
                     "Bruinwalk | Avg GPA | Workload |")
        parts.append("|--------|-------|-------|-------------------|"
                     "----------|---------|----------|")
        for c in top.courses:
            code = c.get("course_code", "")
            title = c.get("title", "")
            units = c.get("units", "")
            enroll = c.get("enrollment_chance")
            enroll_s = f"{enroll:.0%}" if enroll is not None else "N/A"
            bw = c.get("bruinwalk_composite_score")
            bw_s = f"{bw:.2f}" if bw is not None else "N/A"
            gpa = c.get("avg_gpa")
            gpa_s = f"{gpa:.2f}" if gpa is not None else "N/A"
            wl = c.get("workload_hours_per_week")
            wl_s = f"{wl:.1f} hrs" if wl is not None else "N/A"
            parts.append(f"| {code} | {title} | {units} | {enroll_s} | "
                         f"{bw_s} | {gpa_s} | {wl_s} |")
        parts.append("")

        # Day-by-day schedule
        parts.append("### Weekly Schedule")
        for ds in top.day_schedules:
            parts.append(f"**{ds.day}**")
            for sec in ds.sections:
                s_min = sec.get("start_min", 0)
                e_min = sec.get("end_min", 0)
                parts.append(
                    f"- {sec.get('course_code', '')} ({sec.get('section_id', '')}) "
                    f"— {_format_time(s_min)}–{_format_time(e_min)}"
                )
            parts.append("")

        if top.reason:
            parts.append(f"> {top.reason}")
            parts.append("")

    # ---- All 3 Ranked Options ----
    parts.append("## All Ranked Options")
    parts.append("| Rank | Courses | Composite | Enrollment Chance | "
                 "Avg GPA | Schedule Quality |")
    parts.append("|------|---------|-----------|-------------------|"
                 "---------|------------------|")
    for cand in candidates:
        codes = ", ".join(c["course_code"] for c in cand.courses)
        parts.append(
            f"| #{cand.rank} | {codes} | {cand.composite_score:.2f} | "
            f"{cand.avg_enrollment_chance:.0%} | "
            f"{cand.avg_gpa or 'N/A'} | {cand.schedule_quality_score:.2f} |"
        )
    parts.append("")

    # ---- Risk Flags ----
    if flags:
        parts.append("## Risk Flags")
        critical = [f for f in flags if f.severity == "critical"]
        warnings = [f for f in flags if f.severity == "warning"]

        if critical:
            parts.append("### Critical")
            for f in critical:
                parts.append(
                    f"- **{f.course_code}** [{f.flag_type}]: "
                    f"{f.metric_name} = {f.metric_value} "
                    f"(threshold: {f.threshold_used}). {f.explanation}"
                )
            parts.append("")

        if warnings:
            parts.append("### Warnings")
            for f in warnings:
                parts.append(
                    f"- **{f.course_code}** [{f.flag_type}]: "
                    f"{f.metric_name} = {f.metric_value} "
                    f"(threshold: {f.threshold_used}). {f.explanation}"
                )
            parts.append("")
    else:
        parts.append("## Risk Flags")
        parts.append("No risk flags identified.")
        parts.append("")

    # ---- Fallback Suggestions ----
    critical_enrollment = [f for f in flags
                          if f.flag_type == "enrollment" and f.severity == "critical"]
    if critical_enrollment:
        parts.append("## Fallback Suggestions")
        for f in critical_enrollment:
            suggestion = _llm_narrative(
                "You are a UCLA enrollment advisor. Suggest 2 alternative courses "
                "that a student could take instead of the flagged course. Be specific "
                "about course codes and why they are good alternatives.",
                f"Course {f.course_code} has only a {f.metric_value:.0%} chance of "
                f"being open. The student is a {profile.major} {profile.year.value}. "
                f"Suggest 2 alternative courses with higher enrollment availability "
                f"and similar quality.",
                max_tokens=200,
            )
            parts.append(f"**{f.course_code}** (enrollment chance: "
                         f"{f.metric_value:.0%}):")
            parts.append(suggestion)
            parts.append("")
    else:
        parts.append("## Fallback Suggestions")
        parts.append("No critical enrollment risks — no fallbacks needed.")
        parts.append("")

    # ---- Methodology ----
    parts.append("## Methodology")
    parts.append(
        f"Schedules were scored using a weighted composite formula: "
        f"enrollment chance ({profile.weight_enrollment_chance:.0%}), "
        f"professor rating ({profile.weight_professor_rating:.0%}), "
        f"average GPA ({profile.weight_avg_gpa:.0%}), "
        f"schedule quality ({profile.weight_schedule_quality:.0%}), and "
        f"workload balance ({profile.weight_workload:.0%}). "
        f"Each metric was normalised to [0, 1] before weighting. "
        f"Professor ratings were divided by 5.0, GPA by 4.0, and workload "
        f"was inverted (1 − hours/20) so lower workload scores higher. "
        f"The schedule quality sub-score combined average gap time, days on "
        f"campus, and maximum consecutive class minutes. "
        f"Candidates violating hard constraints (time conflicts, day-off "
        f"preferences, time bounds) were filtered before ranking."
    )
    parts.append("")
    parts.append("---")
    parts.append("*Generated by UCLA Course Planner Agent Pipeline*")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agent + Protocol
# ---------------------------------------------------------------------------

agent = Agent(
    name="report-agent",
    seed="report-agent-seed-phrase-change-me",
    port=8008,
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
        "Received %d ranked candidates for %s — building report",
        len(candidates), profile.name,
    )

    # --- Build risk flags ---
    flags = _build_risk_flags(candidates)
    ctx.logger.info("Generated %d risk flags", len(flags))

    # --- Build Markdown report ---
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    markdown = _build_markdown(candidates, profile, flags, now_str)

    # --- Build PlannerReport ---
    top = candidates[0] if candidates else None
    success_prob = top.min_enrollment_chance if top else 0.0

    report = PlannerReport(
        student_name=profile.name,
        generated_at=now_str,
        enrollment_pass=profile.enrollment_pass.value,
        pass_open_datetime=profile.pass_open_datetime,
        recommended_schedule=top,
        all_ranked_schedules=candidates,
        risk_flags=flags,
        full_markdown_report=markdown,
        overall_success_probability=success_prob,
    )

    ctx.logger.info(
        "Report built: %d chars, success prob %.0f%%",
        len(markdown), success_prob * 100,
    )

    # --- Send report back to original user ---
    # The original user's address is threaded through the pipeline
    # in profile.reply_to_user.  If not set, fall back to the
    # input agent address (so the input agent can relay it).
    reply_to = profile.reply_to_user or AGENT_ADDRESSES.get("input", sender)

    await ctx.send(
        reply_to,
        ChatMessage(
            content=[
                TextContent(type="text", text=markdown),
                EndSessionContent(type="end-session"),
            ],
            msg_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
        ),
    )

    ctx.logger.info("Report sent to %s — session ended", reply_to)


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


agent.include(protocol, publish_manifest=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent.run()
