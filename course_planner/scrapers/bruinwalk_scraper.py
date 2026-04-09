"""Scraper for Bruinwalk (bruinwalk.com) professor and course ratings.

Bruinwalk is a React SPA — we try direct HTTP first (looking for embedded
JSON or /api/ endpoints), then fall back to headless Playwright if needed.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from openai import OpenAI

from course_planner.utils import CourseRatings, ProfessorRatings

logger = logging.getLogger(__name__)

BASE_URL = "https://bruinwalk.com"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html, application/json, */*",
}

ASI1_API_KEY = os.environ.get("ASI1_API_KEY", "")
asi_client = OpenAI(base_url="https://api.asi1.ai/v1", api_key=ASI1_API_KEY)

REVIEW_SUMMARY_SYSTEM = (
    "Summarize these UCLA professor reviews in 3 fields: a 1-sentence overall "
    "summary, the most common positive theme, and the most common negative theme. "
    'Return JSON with keys: summary, positive, negative.'
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _slug(name: str) -> str:
    """Convert 'John Smith' -> 'john-smith' for URL paths."""
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def _course_slug(code: str) -> str:
    """Convert 'COM SCI 31' -> 'com-sci-31' for URL paths."""
    return re.sub(r"[^a-z0-9]+", "-", code.strip().lower()).strip("-")


def _extract_float(text: str) -> float | None:
    m = re.search(r"(\d+\.?\d*)", text)
    return float(m.group(1)) if m else None


def _extract_int(text: str) -> int:
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else 0


def _summarize_reviews(reviews: list[str]) -> dict:
    """Use ASI:One to produce summary/positive/negative from raw reviews."""
    if not reviews:
        return {"summary": "", "positive": "", "negative": ""}
    combined = "\n---\n".join(reviews[:30])  # cap to avoid token overflow
    try:
        resp = asi_client.chat.completions.create(
            model="asi1-mini",
            messages=[
                {"role": "system", "content": REVIEW_SUMMARY_SYSTEM},
                {"role": "user", "content": combined},
            ],
            max_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        logger.exception("Review summarization failed")
    return {"summary": "", "positive": "", "negative": ""}


def _try_json_from_page(soup: BeautifulSoup) -> dict | None:
    """Look for __NEXT_DATA__ or similar embedded JSON in a React SPA page."""
    # Next.js pattern
    script = soup.find("script", id="__NEXT_DATA__")
    if script and script.string:
        try:
            return json.loads(script.string)
        except Exception:
            pass

    # Generic: look for large JSON blobs in <script> tags
    for tag in soup.find_all("script"):
        txt = tag.string or ""
        if len(txt) > 200 and ("rating" in txt.lower() or "review" in txt.lower()):
            start = txt.find("{")
            end = txt.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(txt[start:end])
                except Exception:
                    continue
    return None


def _scrape_with_playwright(url: str) -> str:
    """Fallback: use Playwright headless to render a JS-heavy page."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("Playwright not installed — cannot render SPA page")
        return ""

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15_000)
            page.wait_for_load_state("networkidle", timeout=10_000)
            html = page.content()
            browser.close()
            return html
    except Exception:
        logger.exception("Playwright render failed for %s", url)
    return ""


def _extract_reviews_from_soup(soup: BeautifulSoup) -> list[str]:
    """Pull raw review text from the page."""
    reviews: list[str] = []
    # Common selectors for review bodies
    for sel in [
        "div.review-body", "p.review-text", "div.review-content",
        "div[class*='review']", "p[class*='review']",
    ]:
        for el in soup.select(sel):
            text = el.get_text(" ", strip=True)
            if len(text) > 20:
                reviews.append(text)
    if not reviews:
        # Fallback: any <p> inside a reviews container
        container = soup.find(class_=re.compile(r"review", re.I))
        if container:
            for p in container.find_all("p"):
                text = p.get_text(" ", strip=True)
                if len(text) > 20:
                    reviews.append(text)
    return reviews


def _parse_professor_from_json(data: dict, instructor_name: str) -> ProfessorRatings | None:
    """Try to extract professor ratings from embedded JSON data."""
    # Navigate common Next.js / API response shapes
    props = data.get("props", data)
    page_props = props.get("pageProps", props)

    # Look for rating fields in various shapes
    for key in ("professor", "instructor", "data", "result"):
        obj = page_props.get(key, {})
        if isinstance(obj, dict) and ("rating" in str(obj).lower()):
            overall = (
                obj.get("overallRating")
                or obj.get("overall_rating")
                or obj.get("avgRating")
            )
            difficulty = (
                obj.get("difficultyRating")
                or obj.get("difficulty_rating")
                or obj.get("avgDifficulty")
            )
            wta = (
                obj.get("wouldTakeAgainPercent")
                or obj.get("would_take_again_pct")
            )
            total = (
                obj.get("numRatings")
                or obj.get("total_reviews")
                or obj.get("totalReviews")
                or 0
            )
            reviews_raw = obj.get("reviews", [])
            review_texts = []
            for r in reviews_raw:
                if isinstance(r, dict):
                    review_texts.append(r.get("comment", r.get("text", "")))
                elif isinstance(r, str):
                    review_texts.append(r)

            summary = _summarize_reviews(review_texts)

            return ProfessorRatings(
                instructor_name=instructor_name,
                overall_rating=float(overall) if overall else None,
                difficulty_rating=float(difficulty) if difficulty else None,
                would_take_again_pct=float(wta) if wta else None,
                total_reviews=int(total),
                review_summary=summary.get("summary", ""),
                most_common_positive=summary.get("positive", ""),
                most_common_negative=summary.get("negative", ""),
            )
    return None


def _parse_professor_from_html(soup: BeautifulSoup, instructor_name: str) -> ProfessorRatings | None:
    """Try to extract professor ratings from rendered HTML."""
    text = soup.get_text(" ", strip=True)
    if "rating" not in text.lower():
        return None

    overall = None
    difficulty = None
    wta = None
    total = 0

    # Overall rating — look for patterns like "4.2 / 5" or "Overall: 4.2"
    m = re.search(r"(?:overall|rating)[:\s]*(\d+\.?\d*)\s*/?\s*5?", text, re.I)
    if m:
        overall = float(m.group(1))

    m = re.search(r"(?:difficulty)[:\s]*(\d+\.?\d*)", text, re.I)
    if m:
        difficulty = float(m.group(1))

    m = re.search(r"(\d+\.?\d*)\s*%?\s*(?:would take again|WTA)", text, re.I)
    if m:
        wta = float(m.group(1))

    m = re.search(r"(\d+)\s*(?:review|rating)s?", text, re.I)
    if m:
        total = int(m.group(1))

    reviews = _extract_reviews_from_soup(soup)
    summary = _summarize_reviews(reviews)

    if overall is not None or total > 0:
        return ProfessorRatings(
            instructor_name=instructor_name,
            overall_rating=overall,
            difficulty_rating=difficulty,
            would_take_again_pct=wta,
            total_reviews=total,
            review_summary=summary.get("summary", ""),
            most_common_positive=summary.get("positive", ""),
            most_common_negative=summary.get("negative", ""),
        )
    return None


def _parse_course_from_json(data: dict, course_code: str) -> CourseRatings | None:
    """Try to extract course ratings from embedded JSON data."""
    props = data.get("props", data)
    page_props = props.get("pageProps", props)

    for key in ("course", "data", "result"):
        obj = page_props.get(key, {})
        if isinstance(obj, dict) and ("rating" in str(obj).lower() or "review" in str(obj).lower()):
            overall = (
                obj.get("overallRating")
                or obj.get("overall_rating")
                or obj.get("avgRating")
            )
            total = obj.get("numRatings") or obj.get("total_reviews") or 0
            hours = obj.get("avgHoursPerWeek") or obj.get("hours_per_week")
            grade = obj.get("avgGrade") or obj.get("average_grade") or ""
            if overall is not None:
                return CourseRatings(
                    course_code=course_code,
                    overall_course_rating=float(overall),
                    total_reviews=int(total),
                    avg_hours_per_week=float(hours) if hours else None,
                    avg_grade_expected=str(grade),
                )
    return None


def _parse_course_from_html(soup: BeautifulSoup, course_code: str) -> CourseRatings | None:
    """Try to extract course ratings from rendered HTML."""
    text = soup.get_text(" ", strip=True)
    overall = None
    total = 0
    hours = None
    grade = ""

    m = re.search(r"(?:overall|course\s*rating)[:\s]*(\d+\.?\d*)", text, re.I)
    if m:
        overall = float(m.group(1))

    m = re.search(r"(\d+)\s*(?:review|rating)s?", text, re.I)
    if m:
        total = int(m.group(1))

    m = re.search(r"(\d+\.?\d*)\s*(?:hours?|hrs?)\s*(?:per|/)\s*week", text, re.I)
    if m:
        hours = float(m.group(1))

    m = re.search(r"(?:average|avg)\s*grade[:\s]*([A-F][+-]?)", text, re.I)
    if m:
        grade = m.group(1)

    if overall is not None or total > 0:
        return CourseRatings(
            course_code=course_code,
            overall_course_rating=overall,
            total_reviews=total,
            avg_hours_per_week=hours,
            avg_grade_expected=grade,
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scrape_professor_ratings(
    instructor_name: str, course_code: str
) -> Optional[ProfessorRatings]:
    """Search Bruinwalk for *instructor_name* and return their ratings, or None."""
    if not instructor_name.strip():
        return None

    slug = _slug(instructor_name)
    url = f"{BASE_URL}/professors/{slug}/"

    try:
        with httpx.Client(headers=_HEADERS, timeout=20, follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                # Try search endpoint
                search_resp = client.get(
                    f"{BASE_URL}/search/",
                    params={"q": instructor_name, "type": "professor"},
                )
                if search_resp.status_code != 200:
                    return None
                soup = BeautifulSoup(search_resp.text, "html.parser")
                # Find first professor link
                link = soup.find("a", href=re.compile(r"/professors/"))
                if not link:
                    return None
                prof_url = BASE_URL + link["href"]
                resp = client.get(prof_url)
                if resp.status_code != 200:
                    return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # Try embedded JSON first
            json_data = _try_json_from_page(soup)
            if json_data:
                result = _parse_professor_from_json(json_data, instructor_name)
                if result:
                    return result

            # Try HTML parsing
            result = _parse_professor_from_html(soup, instructor_name)
            if result:
                return result

            # Try /api/ endpoint
            api_resp = client.get(
                f"{BASE_URL}/api/professors/{slug}/",
                headers={**_HEADERS, "Accept": "application/json"},
            )
            if api_resp.status_code == 200:
                try:
                    api_data = api_resp.json()
                    result = _parse_professor_from_json(
                        {"props": {"pageProps": {"professor": api_data}}},
                        instructor_name,
                    )
                    if result:
                        return result
                except Exception:
                    pass

    except Exception:
        logger.exception("Professor scrape failed for %s", instructor_name)

    # Playwright fallback
    html = _scrape_with_playwright(url)
    if html:
        soup = BeautifulSoup(html, "html.parser")
        json_data = _try_json_from_page(soup)
        if json_data:
            result = _parse_professor_from_json(json_data, instructor_name)
            if result:
                return result
        return _parse_professor_from_html(soup, instructor_name)

    return None


def scrape_course_ratings(course_code: str) -> Optional[CourseRatings]:
    """Scrape the course page on Bruinwalk and return ratings, or None."""
    if not course_code.strip():
        return None

    slug = _course_slug(course_code)
    url = f"{BASE_URL}/classes/{slug}/"

    try:
        with httpx.Client(headers=_HEADERS, timeout=20, follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                # Try search
                search_resp = client.get(
                    f"{BASE_URL}/search/",
                    params={"q": course_code, "type": "course"},
                )
                if search_resp.status_code != 200:
                    return None
                soup = BeautifulSoup(search_resp.text, "html.parser")
                link = soup.find("a", href=re.compile(r"/classes/"))
                if not link:
                    return None
                course_url = BASE_URL + link["href"]
                resp = client.get(course_url)
                if resp.status_code != 200:
                    return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # Try embedded JSON
            json_data = _try_json_from_page(soup)
            if json_data:
                result = _parse_course_from_json(json_data, course_code)
                if result:
                    return result

            # Try HTML
            result = _parse_course_from_html(soup, course_code)
            if result:
                return result

            # Try /api/ endpoint
            api_resp = client.get(
                f"{BASE_URL}/api/classes/{slug}/",
                headers={**_HEADERS, "Accept": "application/json"},
            )
            if api_resp.status_code == 200:
                try:
                    api_data = api_resp.json()
                    result = _parse_course_from_json(
                        {"props": {"pageProps": {"course": api_data}}},
                        course_code,
                    )
                    if result:
                        return result
                except Exception:
                    pass

    except Exception:
        logger.exception("Course scrape failed for %s", course_code)

    # Playwright fallback
    html = _scrape_with_playwright(url)
    if html:
        soup = BeautifulSoup(html, "html.parser")
        json_data = _try_json_from_page(soup)
        if json_data:
            result = _parse_course_from_json(json_data, course_code)
            if result:
                return result
        return _parse_course_from_html(soup, course_code)

    return None
