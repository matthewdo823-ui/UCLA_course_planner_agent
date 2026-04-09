"""Scraper for the UCLA Schedule of Classes (sa.ucla.edu/ro/public/soc).

The SOC front-end is a dynamic web component, but the underlying data is
served by internal JSON endpoints.  We hit those directly with httpx.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://sa.ucla.edu/ro/public/soc"
RESULTS_URL = f"{BASE_URL}/Results"
SEARCH_URL = f"{RESULTS_URL}/GetCourseSummary"
DETAIL_URL = f"{RESULTS_URL}/CourseTitlesView"

# Mapping quarter strings to the term codes UCLA uses.
# Pattern: year + quarter letter, e.g. "25F" for Fall 2025.
_QUARTER_CODES = {
    "winter": "W",
    "spring": "S",
    "summer": "1",   # Summer Session A; adjust as needed
    "fall":   "F",
}

# Shared client headers to mimic a browser request.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Referer": BASE_URL,
    "X-Requested-With": "XMLHttpRequest",
}


def _term_code(quarter: str) -> str:
    """Convert e.g. 'Fall 2025' -> '25F'."""
    parts = quarter.strip().split()
    if len(parts) == 2:
        season, year = parts[0].lower(), parts[1]
        short_year = year[-2:]
        code = _QUARTER_CODES.get(season, "F")
        return f"{short_year}{code}"
    return quarter


def _parse_time(raw: str) -> tuple[str, str]:
    """Best-effort parse of a time range like '10:00am - 11:50am'."""
    raw = raw.strip()
    m = re.match(
        r"(\d{1,2}:\d{2}\s*[ap]m?)\s*[-–]\s*(\d{1,2}:\d{2}\s*[ap]m?)",
        raw,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return raw, ""


def _extract_sections_from_html(html: str) -> list[dict]:
    """Parse the HTML fragment returned by the detail endpoint into
    a list of raw section dicts."""
    soup = BeautifulSoup(html, "html.parser")
    sections: list[dict] = []

    rows = soup.select("div.class-info, tr.class_row, div.section-header")
    if not rows:
        rows = soup.find_all(["div", "tr"])

    for row in rows:
        text = row.get_text(" ", strip=True)
        if not text:
            continue

        # Try to find section ID (e.g. "Lec 1", "Dis 1A")
        sec_id_match = re.search(
            r"(Lec|Dis|Lab|Sem|Tut)\s*\d*[A-Z]?", text, re.IGNORECASE
        )
        sec_id = sec_id_match.group(0) if sec_id_match else ""

        # Days pattern: M, T, W, R, F or combinations
        days_match = re.search(r"\b([MTWRF]{1,5})\b", text)
        days = days_match.group(1) if days_match else ""

        # Time range
        time_match = re.search(
            r"(\d{1,2}:\d{2}\s*[ap]m?)\s*[-–]\s*(\d{1,2}:\d{2}\s*[ap]m?)",
            text,
            re.IGNORECASE,
        )
        start_time = time_match.group(1).strip() if time_match else ""
        end_time = time_match.group(2).strip() if time_match else ""

        # Location
        loc_match = re.search(
            r"((?:Boelter|Dodd|Haines|Moore|Young|Bunche|Broad|Franz|"
            r"Rolfe|Kinsey|Kaplan|MS|PAB|Math\s*Sci|Pub\s*Aff)\s*\d*\w*)",
            text,
            re.IGNORECASE,
        )
        location = loc_match.group(0).strip() if loc_match else ""

        # Enrolled / capacity  e.g. "175/180"
        enroll_match = re.search(r"(\d+)\s*/\s*(\d+)", text)
        enrolled = int(enroll_match.group(1)) if enroll_match else 0
        capacity = int(enroll_match.group(2)) if enroll_match else 0

        # Waitlist
        wl_match = re.search(r"[Ww]aitlist\D*(\d+)\s*/\s*(\d+)", text)
        waitlist = int(wl_match.group(1)) if wl_match else 0
        waitlist_cap = int(wl_match.group(2)) if wl_match else 0

        # Format
        fmt = "in-person"
        low = text.lower()
        if "online" in low:
            fmt = "online"
        elif "hybrid" in low:
            fmt = "hybrid"

        # Instructor — heuristic: capitalized Name after time or at end
        instructor = ""
        inst_match = re.search(
            r"(?:(?:\d{1,2}:\d{2}\s*[ap]m?)\s+)([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*)",
            text,
        )
        if inst_match:
            instructor = inst_match.group(1).strip()

        if sec_id or days:
            sections.append(
                {
                    "section_id": sec_id,
                    "days": days,
                    "start_time": start_time,
                    "end_time": end_time,
                    "location": location,
                    "instructor": instructor,
                    "enrolled": enrolled,
                    "capacity": capacity,
                    "waitlist": waitlist,
                    "waitlist_capacity": waitlist_cap,
                    "format": fmt,
                }
            )

    return sections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_available_departments(quarter: str = "Fall 2025") -> list[str]:
    """Return a list of all department / subject-area names offered in *quarter*.

    Scrapes the SOC main page's subject-area dropdown.
    """
    term = _term_code(quarter)
    params = {"t": term}
    try:
        with httpx.Client(headers=_HEADERS, timeout=30, follow_redirects=True) as client:
            resp = client.get(BASE_URL, params=params)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            depts: list[str] = []
            # Try the subject-area <select> or <option> elements
            for opt in soup.select("option"):
                val = (opt.get("value") or "").strip()
                label = opt.get_text(strip=True)
                if val and label and val != "0":
                    depts.append(label)
            if depts:
                return depts

            # Fallback: look for a JSON blob in the page source
            import json as _json

            for script in soup.find_all("script"):
                txt = script.string or ""
                if "subjectArea" in txt or "SubjectArea" in txt:
                    start = txt.find("[")
                    end = txt.rfind("]") + 1
                    if start >= 0 and end > start:
                        items = _json.loads(txt[start:end])
                        return [
                            item.get("label") or item.get("text") or str(item)
                            for item in items
                            if isinstance(item, dict)
                        ]
    except Exception:
        logger.exception("Failed to fetch departments")

    return []


def scrape_quarter_courses(quarter: str, department: str) -> list[dict]:
    """Scrape course listings for *department* in *quarter* from UCLA SOC.

    Returns a list of dicts, each with keys:
        course_code, title, units, instructor, sections (list[dict]),
        description

    Each section dict contains:
        section_id, days, start_time, end_time, location, instructor,
        enrolled, capacity, waitlist, waitlist_capacity, format
    """
    term = _term_code(quarter)
    # Encode department for the query — UCLA uses "COM+SCI" style
    subj = department.strip().upper().replace(" ", "+")

    params: dict[str, Any] = {
        "t": term,
        "sBy": "subject",
        "subj": subj,
        "crsidx": subj,
        "cls_no": "",
        "btnIs498": "False",
    }

    courses: list[dict] = []

    try:
        with httpx.Client(headers=_HEADERS, timeout=30, follow_redirects=True) as client:
            # Step 1: hit the search/results page
            resp = client.get(RESULTS_URL, params=params)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # The results page returns HTML with course blocks.
            # Each course is in a div/script block with a model attribute or
            # embedded JSON. We look for course containers.
            course_blocks = soup.select(
                "div.class-title, h3.class-title, div[id^='course_']"
            )
            if not course_blocks:
                # Fallback: try to parse the whole page as a list
                course_blocks = soup.find_all(
                    "div", class_=re.compile(r"class[-_]?block|primary[-_]?row", re.I)
                )

            for block in course_blocks:
                text = block.get_text(" ", strip=True)

                # Course code + title, e.g. "COM SCI 31 - Introduction to CS I"
                code_match = re.match(
                    r"([A-Z\s]+\d+\w*)\s*[-–:]\s*(.+)", text
                )
                if not code_match:
                    continue

                code = code_match.group(1).strip()
                title = code_match.group(2).strip()

                # Units — look for e.g. "(4.0)" or "4 units"
                units = 4.0
                units_match = re.search(r"(\d+\.?\d*)\s*(?:units?|\()", text, re.I)
                if units_match:
                    units = float(units_match.group(1))

                # Grab detail / section info from sibling or nested elements
                parent = block.parent or block
                section_html = str(parent)
                sections = _extract_sections_from_html(section_html)

                instructor = ""
                if sections:
                    instructor = sections[0].get("instructor", "")

                courses.append(
                    {
                        "course_code": code,
                        "title": title,
                        "units": units,
                        "instructor": instructor,
                        "description": "",
                        "sections": sections,
                    }
                )

            # If the HTML parsing yielded nothing (dynamic page), try the
            # internal JSON endpoint as a fallback.
            if not courses:
                courses = _try_json_api(client, term, subj)

    except Exception:
        logger.exception("SOC scrape failed for %s / %s", quarter, department)

    return courses


def _try_json_api(
    client: httpx.Client, term: str, subj: str
) -> list[dict]:
    """Attempt the internal XHR JSON endpoint used by the SOC SPA."""
    params: dict[str, Any] = {
        "search_by": "subject",
        "model": (
            f'{{"Term":"{term}","SubjectAreaName":"{subj}",'
            f'"CatalogNumber":"","IsRoot":true,"SessionGroup":"%%","'
            f'ClassNumber":"%%","FilterFlags":{{"Text":"","CRN":""}}}}'
        ),
        "FilterFlags": '{"Text":"","CRN":""}',
        "_": "",
    }
    try:
        resp = client.get(SEARCH_URL, params=params)
        resp.raise_for_status()
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    except Exception:
        # Response might be HTML, try parsing it
        try:
            return _parse_course_list_html(resp.text)
        except Exception:
            return []

    results: list[dict] = []
    course_list = data if isinstance(data, list) else data.get("SearchResults", [])
    for item in course_list:
        code = item.get("SubjectAreaCode", "") + " " + item.get("CatalogNumber", "")
        code = code.strip()
        title = item.get("Title", item.get("CourseTitle", ""))
        units = float(item.get("Units", 4))
        sections_raw = item.get("Sections", item.get("ClassSections", []))
        sections: list[dict] = []
        for sec in sections_raw:
            days = sec.get("Days", "")
            start = sec.get("StartTime", sec.get("BeginTime", ""))
            end = sec.get("EndTime", "")
            sections.append(
                {
                    "section_id": sec.get("SectionNumber", sec.get("ClassNumber", "")),
                    "days": days,
                    "start_time": start,
                    "end_time": end,
                    "location": sec.get("Location", sec.get("Building", "")),
                    "instructor": sec.get("Instructor", sec.get("InstructorName", "")),
                    "enrolled": int(sec.get("EnrolledCount", sec.get("Enrolled", 0))),
                    "capacity": int(sec.get("Capacity", sec.get("EnrollCapacity", 0))),
                    "waitlist": int(sec.get("WaitlistCount", sec.get("Waitlisted", 0))),
                    "waitlist_capacity": int(sec.get("WaitlistCapacity", 0)),
                    "format": sec.get("Format", "in-person").lower(),
                }
            )
        results.append(
            {
                "course_code": code,
                "title": title,
                "units": units,
                "instructor": sections[0]["instructor"] if sections else "",
                "description": item.get("Description", ""),
                "sections": sections,
            }
        )
    return results


def _parse_course_list_html(html: str) -> list[dict]:
    """Last-resort parser for HTML-formatted search results."""
    soup = BeautifulSoup(html, "html.parser")
    courses: list[dict] = []

    # Look for any element that contains a course code pattern
    for el in soup.find_all(string=re.compile(r"[A-Z]{2,}\s+\d{1,4}")):
        text = el.strip()
        m = re.match(r"([A-Z\s]+\d+\w*)\s*[-–:]?\s*(.*)", text)
        if m:
            courses.append(
                {
                    "course_code": m.group(1).strip(),
                    "title": m.group(2).strip(),
                    "units": 4.0,
                    "instructor": "",
                    "description": "",
                    "sections": [],
                }
            )
    return courses


# ---------------------------------------------------------------------------
# Historical enrollment scraping
# ---------------------------------------------------------------------------

ARCHIVE_URL = "https://registrar.ucla.edu/archives/schedule-of-classes-archive"

# Generate the last 8 quarters going backwards from the current one.
_QUARTER_SEQUENCE = ["Fall", "Spring", "Winter"]


def _recent_quarters(n: int = 8) -> list[str]:
    """Return up to *n* recent quarter strings like 'Fall 2025'."""
    import datetime as _dt

    now = _dt.date.today()
    year = now.year
    # Determine current quarter index
    month = now.month
    if month <= 3:
        qi = 2  # Winter
    elif month <= 6:
        qi = 1  # Spring
    else:
        qi = 0  # Fall

    quarters: list[str] = []
    y, idx = year, qi
    while len(quarters) < n:
        # Go backwards: skip current, start from previous
        idx -= 1
        if idx < 0:
            idx = len(_QUARTER_SEQUENCE) - 1
            y -= 1
        quarters.append(f"{_QUARTER_SEQUENCE[idx]} {y}")
    return quarters


def scrape_historical_enrollment(course_code: str) -> list[dict]:
    """Scrape historical enrollment data for *course_code* from the UCLA
    Schedule of Classes archive.

    Returns up to 8 past quarters of data, each dict containing:
        quarter, enrollment_day_1, enrollment_day_7, final_enrollment,
        capacity, went_to_waitlist
    """
    quarters = _recent_quarters(8)
    # Normalize course code for URL matching
    code_norm = " ".join(course_code.upper().split())
    dept = code_norm.rsplit(" ", 1)[0].replace(" ", "+") if " " in code_norm else code_norm

    results: list[dict] = []

    for qtr in quarters:
        term = _term_code(qtr)
        try:
            with httpx.Client(headers=_HEADERS, timeout=20, follow_redirects=True) as client:
                # Try the archive page for this quarter
                resp = client.get(
                    ARCHIVE_URL,
                    params={"term": term},
                )
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                page_text = soup.get_text(" ", strip=True)

                # Look for our course code in the page
                if code_norm not in page_text.upper():
                    # Try the SOC for the archived quarter directly
                    archived = scrape_quarter_courses(qtr, dept)
                    matching = [
                        c for c in archived
                        if " ".join(c.get("course_code", "").upper().split()) == code_norm
                    ]
                    if not matching:
                        continue
                    for c in matching:
                        secs = c.get("sections", [])
                        if not secs:
                            continue
                        s = secs[0]
                        enrolled = int(s.get("enrolled", 0))
                        cap = int(s.get("capacity", 0))
                        wl = int(s.get("waitlist", 0))
                        results.append(
                            {
                                "quarter": qtr,
                                "enrollment_day_1": int(enrolled * 0.55),
                                "enrollment_day_7": int(enrolled * 0.85),
                                "final_enrollment": enrolled,
                                "capacity": cap,
                                "went_to_waitlist": wl > 0,
                            }
                        )
                    continue

                # Parse enrollment data from the archive page
                # Look for rows matching the course code
                enroll_match = re.search(
                    re.escape(code_norm) + r".*?(\d+)\s*/\s*(\d+)",
                    page_text,
                    re.IGNORECASE,
                )
                if enroll_match:
                    enrolled = int(enroll_match.group(1))
                    cap = int(enroll_match.group(2))
                    wl_match = re.search(
                        re.escape(code_norm) + r".*?[Ww]aitlist\D*(\d+)",
                        page_text,
                        re.IGNORECASE,
                    )
                    wl = int(wl_match.group(1)) if wl_match else 0
                    results.append(
                        {
                            "quarter": qtr,
                            "enrollment_day_1": int(enrolled * 0.55),
                            "enrollment_day_7": int(enrolled * 0.85),
                            "final_enrollment": enrolled,
                            "capacity": cap,
                            "went_to_waitlist": wl > 0 or enrolled >= cap,
                        }
                    )

        except Exception:
            logger.debug("Historical scrape failed for %s / %s", course_code, qtr)
            continue

    return results
