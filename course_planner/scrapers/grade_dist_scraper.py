"""Grade-distribution scraper — downloads UCLA grade data from public Google
Sheets, parses with pandas, and returns a nested lookup dict.

Data is loaded and cached once at import time (module-level) so that
individual agent requests don't re-download.
"""

from __future__ import annotations

import io
import logging
import math
from collections import defaultdict

import httpx
import pandas as pd

from course_planner.utils import GradeDistribution

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google Sheets — export-as-CSV URLs
# ---------------------------------------------------------------------------

_SHEETS: list[dict[str, str]] = [
    {
        "label": "Fall 2021–Summer 2022",
        "url": (
            "https://docs.google.com/spreadsheets/d/"
            "1qIxmVb590IoAm8hcb7Pjp3C5SQeyt5Hz/export?format=csv&gid=787093573"
        ),
    },
    {
        "label": "Fall 2022–Spring 2023",
        "url": (
            "https://docs.google.com/spreadsheets/d/"
            "1Nb50cKvHmfd3pelk-FAw2C1yU7QFuFm8/export?format=csv"
        ),
    },
    {
        "label": "Summer 2023–Spring 2024",
        "url": (
            "https://docs.google.com/spreadsheets/d/"
            "10FY4R340AC-vRrFVbVXYuna85Um9z5EO/export?format=csv"
        ),
    },
    {
        "label": "Summer 2024–Spring 2025",
        "url": (
            "https://docs.google.com/spreadsheets/d/"
            "1-4xWiDA4jQYkItifWtDVUZrVhVXdYuyo/export?format=csv"
        ),
    },
]

# GPA points for each letter grade
_GPA_MAP: dict[str, float] = {
    "A+": 4.0, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "D-": 0.7,
    "F": 0.0,
}

# Map from grade letter to the count column name we look for in the CSV.
# UCLA sheets typically have columns like "A+", "A", "A-", "B+", …
_GRADE_COLS = ["A+", "A", "A-", "B+", "B", "B-",
               "C+", "C", "C-", "D+", "D", "D-", "F"]


# ---------------------------------------------------------------------------
# Column-name normalisation
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first column from *candidates* that exists (case-insensitive)."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        hit = lower_map.get(c.lower().strip())
        if hit is not None:
            return hit
    return None


def _safe_int(val) -> int:
    try:
        v = float(val)
        return int(v) if not math.isnan(v) else 0
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Core: download + parse all sheets
# ---------------------------------------------------------------------------

# Each row becomes a _RawRow for aggregation.
class _RawRow:
    __slots__ = (
        "course_code", "instructor", "quarter",
        "counts",  # dict[str, int] — grade -> count
    )

    def __init__(self, course_code: str, instructor: str, quarter: str,
                 counts: dict[str, int]):
        self.course_code = course_code
        self.instructor = instructor
        self.quarter = quarter
        self.counts = counts


def _download_sheet(url: str, label: str) -> pd.DataFrame:
    """Download a Google Sheet as CSV and return a DataFrame."""
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.text))
    except Exception:
        logger.exception("Failed to download sheet: %s", label)
        return pd.DataFrame()


def _parse_rows(df: pd.DataFrame, label: str) -> list[_RawRow]:
    """Extract _RawRow objects from a DataFrame.

    The UCLA grade sheets store one row per *grade* per section, e.g.:
        Sheet format A (newer):  subj_area_cd | disp_catlg_no | grd_cd | num_grd | instr_nm | enrl_term_cd
        Sheet format B (older):  SUBJECT AREA | CATLG NBR     | GRD OFF | GRD COUNT | INSTR NAME | ENROLLMENT TERM

    We pivot so each _RawRow has the full grade-count dict for one
    (course, instructor, quarter) combination.
    """
    if df.empty:
        return []

    # --- Detect column names (two known schemas) ---
    subj_col = _find_col(df, "subj_area_cd", "subj_area_name", "SUBJECT AREA")
    catalog_col = _find_col(df, "disp_catlg_no", "crs_catlg_no", "CATLG NBR")
    grade_col = _find_col(df, "grd_cd", "GRD OFF")
    count_col = _find_col(df, "num_grd", "GRD COUNT")
    instr_col = _find_col(df, "instr_nm", "INSTR NAME")
    term_col = _find_col(df, "enrl_term_cd", "ENROLLMENT TERM")

    if not (subj_col and catalog_col and grade_col and count_col):
        logger.warning(
            "Could not identify required columns in sheet %s. Columns: %s",
            label, list(df.columns),
        )
        return []

    # --- Group rows by (course_code, instructor, quarter) and pivot grades ---
    grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        subj = str(row.get(subj_col, "")).strip()
        cat = str(row.get(catalog_col, "")).strip()
        if not subj or not cat:
            continue
        code = f"{subj} {cat}".upper()
        code = " ".join(code.split())  # normalise whitespace

        instructor = str(row.get(instr_col, "")).strip() if instr_col else ""
        quarter = str(row.get(term_col, label)).strip() if term_col else label

        grade = str(row.get(grade_col, "")).strip().upper()
        # Normalise grade strings: "A+" / "A +" -> "A+"
        grade = grade.replace(" ", "")
        count = _safe_int(row.get(count_col, 0))

        if grade in _GPA_MAP or grade in ("D-",):  # D- isn't in _GPA_MAP but is valid
            key = (code, instructor, quarter)
            grouped[key][grade] += count

    # --- Convert to _RawRow list ---
    rows: list[_RawRow] = []
    for (code, instructor, quarter), counts in grouped.items():
        rows.append(_RawRow(code, instructor, quarter, dict(counts)))

    return rows


# ---------------------------------------------------------------------------
# Aggregation → GradeDistribution
# ---------------------------------------------------------------------------

def _aggregate(rows: list[_RawRow]) -> GradeDistribution:
    """Aggregate multiple _RawRow objects into one GradeDistribution."""
    if not rows:
        return GradeDistribution(course_code="", instructor_name="")

    course_code = rows[0].course_code
    instructor = rows[0].instructor

    # Sum all count fields
    totals: dict[str, int] = defaultdict(int)
    for r in rows:
        for g, c in r.counts.items():
            totals[g] += c

    count_a_plus = totals.get("A+", 0)
    count_a = totals.get("A", 0)
    count_a_minus = totals.get("A-", 0)
    count_b_plus = totals.get("B+", 0)
    count_b = totals.get("B", 0)
    count_b_minus = totals.get("B-", 0)
    count_c_plus = totals.get("C+", 0)
    count_c = totals.get("C", 0)
    count_c_minus = totals.get("C-", 0)
    count_d_plus = totals.get("D+", 0)
    count_d = totals.get("D", 0)
    count_d_minus = totals.get("D-", 0)
    count_f = totals.get("F", 0)

    total_students = sum(totals.values())

    # Expand to per-student GPA list for avg/median/std
    gpa_values: list[float] = []
    for g, cnt in totals.items():
        gpa = _GPA_MAP.get(g)
        if gpa is not None:
            gpa_values.extend([gpa] * cnt)

    avg_gpa = sum(gpa_values) / len(gpa_values) if gpa_values else 0.0
    sorted_gpas = sorted(gpa_values)
    n = len(sorted_gpas)
    median_gpa = 0.0
    if n > 0:
        mid = n // 2
        median_gpa = (sorted_gpas[mid] if n % 2 == 1
                      else (sorted_gpas[mid - 1] + sorted_gpas[mid]) / 2)

    std_dev_gpa = 0.0
    if n > 1:
        variance = sum((g - avg_gpa) ** 2 for g in gpa_values) / (n - 1)
        std_dev_gpa = math.sqrt(variance)

    # Percentages
    t = total_students or 1
    a_range = count_a_plus + count_a + count_a_minus
    b_range = count_b_plus + count_b + count_b_minus
    c_range = count_c_plus + count_c + count_c_minus
    d_or_f = count_d_plus + count_d + count_d_minus + count_f

    # GPA trend (linear slope across quarters)
    # Compute per-quarter avg_gpa, then regress on index
    quarter_gpas: list[float] = []
    for r in rows:
        row_vals: list[float] = []
        for g, cnt in r.counts.items():
            gpa = _GPA_MAP.get(g)
            if gpa is not None:
                row_vals.extend([gpa] * cnt)
        if row_vals:
            quarter_gpas.append(sum(row_vals) / len(row_vals))

    gpa_trend = 0.0
    if len(quarter_gpas) >= 2:
        n_q = len(quarter_gpas)
        x_mean = (n_q - 1) / 2
        y_mean = sum(quarter_gpas) / n_q
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(quarter_gpas))
        den = sum((i - x_mean) ** 2 for i in range(n_q))
        if den > 0:
            gpa_trend = num / den

    # Most recent quarter avg
    most_recent_avg = quarter_gpas[-1] if quarter_gpas else avg_gpa

    return GradeDistribution(
        course_code=course_code,
        instructor_name=instructor,
        count_a_plus=count_a_plus,
        count_a=count_a,
        count_a_minus=count_a_minus,
        count_b_plus=count_b_plus,
        count_b=count_b,
        count_b_minus=count_b_minus,
        count_c_plus=count_c_plus,
        count_c=count_c,
        count_c_minus=count_c_minus,
        count_d_plus=count_d_plus,
        count_d=count_d,
        count_d_minus=count_d_minus,
        count_f=count_f,
        total_students=total_students,
        avg_gpa=round(avg_gpa, 3),
        median_gpa=round(median_gpa, 3),
        std_dev_gpa=round(std_dev_gpa, 3),
        pct_a_range=round(a_range / t, 4),
        pct_b_range=round(b_range / t, 4),
        pct_c_range=round(c_range / t, 4),
        pct_d_or_f=round(d_or_f / t, 4),
        gpa_trend=round(gpa_trend, 4),
        most_recent_quarter_avg_gpa=round(most_recent_avg, 3),
        quarters_sampled=len(rows),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Module-level cache — populated on first call to load_grade_data().
_CACHE: dict[str, dict[str, GradeDistribution]] | None = None


def load_grade_data() -> dict[str, dict[str, GradeDistribution]]:
    """Download all four Google Sheets, parse, aggregate, and return
    ``{ course_code: { instructor_name: GradeDistribution } }``.

    Results are cached after the first call.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    logger.info("Loading grade distribution data from %d sheets …", len(_SHEETS))

    all_rows: list[_RawRow] = []
    for sheet in _SHEETS:
        df = _download_sheet(sheet["url"], sheet["label"])
        parsed = _parse_rows(df, sheet["label"])
        logger.info("  %s → %d rows", sheet["label"], len(parsed))
        all_rows.extend(parsed)

    logger.info("Total raw rows: %d", len(all_rows))

    # Group by (course_code, instructor)
    grouped: dict[tuple[str, str], list[_RawRow]] = defaultdict(list)
    for r in all_rows:
        key = (" ".join(r.course_code.upper().split()),
               r.instructor.strip())
        grouped[key].append(r)

    # Aggregate
    result: dict[str, dict[str, GradeDistribution]] = defaultdict(dict)
    for (code, instructor), rows in grouped.items():
        dist = _aggregate(rows)
        result[code][instructor] = dist

    _CACHE = dict(result)
    logger.info("Grade data loaded: %d courses", len(_CACHE))
    return _CACHE
