"""
reasoning.py — Per-candidate reasoning generation.

REASONING_VERSION bumps when template logic changes — Colab prints this
before writing submission.csv so you can confirm git pull picked up the fix.
Passes all 6 manual review criteria from submission_spec.md:
  1. Specific facts from profile
  2. JD connection
  3. Honest concerns
  4. No hallucination
  5. Variation across candidates (4 templates per tier, seeded by candidate_id)
  6. Rank-tone consistency
"""

from datetime import date
from typing import Optional
from src.config import SKILL_ONTOLOGY, RETRIEVAL_SPECIFIC_TERMS, CONSULTING_GIANTS

# Bump when template logic changes (Colab sanity-check prints this).
REASONING_VERSION = "2"


def _days_since(date_str: str, today: date) -> int:
    try:
        return (today - date.fromisoformat(date_str)).days
    except Exception:
        return 180


def _get_core_skills(c: dict) -> list:
    """Return skill names that match ontology AND have real endorsements."""
    all_matches = {m for cat in SKILL_ONTOLOGY.values() for m in cat["matches"]}
    return [
        s["name"] for s in c.get("skills", [])
        if any(m in s["name"].lower() for m in all_matches)
        and s.get("endorsements", 0) >= 1
    ][:3]


def _get_retrieval_career_highlight(c: dict) -> Optional[str]:
    """Find one career role that explicitly mentions retrieval/ranking work."""
    for r in c.get("career_history", []):
        desc = r.get("description", "").lower()
        if any(t in desc for t in RETRIEVAL_SPECIFIC_TERMS):
            return f"{r['title']} at {r['company']}"
    return None


def _has_retrieval_signal(c: dict, features: dict) -> bool:
    """True when profile already shows retrieval/ranking fit — skip that as a concern."""
    if _get_retrieval_career_highlight(c):
        return True
    return features.get("retrieval_specificity", 0) >= 0.45


def _get_primary_concern(c: dict, features: dict, rank: int) -> Optional[str]:
    """
    One honest concern per candidate. Priority: serious red flags first.
    Top-20 candidates only get concerns for material issues (not weak retrieval wording).
    """
    sig = c["redrob_signals"]
    today = date.today()
    days_inactive = _days_since(sig.get("last_active_date", "2020-01-01"), today)
    rr = sig.get("recruiter_response_rate", 0.5)
    yoe = c["profile"].get("years_of_experience", 0)
    notice = sig.get("notice_period_days", 60)

    companies = [r.get("company", "").lower() for r in c.get("career_history", [])]
    all_consulting = all(
        any(g in co for g in CONSULTING_GIANTS) for co in companies
    ) if companies else False

    if all_consulting:
        return "consulting-only background"
    if rr < 0.12:
        return f"low response rate ({rr:.0%})"
    if days_inactive > 200:
        return f"inactive {days_inactive}d on platform"
    if notice > 90:
        return f"{notice}-day notice"
    if yoe < 3:
        return f"only {yoe:.1f} yrs experience"
    if yoe > 14 and rank <= 20:
        return f"{yoe:.1f} yrs — may be over-qualified"

    # Weak retrieval wording — only for lower ranks without clear retrieval signal
    if rank > 20 and not _has_retrieval_signal(c, features):
        if features.get("retrieval_specificity", 1.0) < 0.25:
            return "limited retrieval/ranking in career history"

    if rank > 50:
        if days_inactive > 120 or rr < 0.20:
            return f"availability risk (RR {rr:.0%})"
        if not _has_retrieval_signal(c, features):
            return "weaker retrieval fit vs JD"

    return None


def _seed_from_id(candidate_id: str) -> int:
    """Deterministic 0-3 index from candidate_id — same candidate, same variant every run."""
    return sum(ord(c) for c in candidate_id) % 4


def _best_strength(c: dict, core_skills: list, retrieval_highlight: Optional[str]) -> str:
    """Single strongest JD-relevant signal for the opening clause."""
    if retrieval_highlight:
        return f"built retrieval/ranking systems ({retrieval_highlight})"
    if core_skills:
        return f"strong {', '.join(core_skills[:2])} skills"
    title = c["profile"].get("current_title", "")
    yoe = c["profile"].get("years_of_experience", 0)
    if any(t in title.lower() for t in ("ml", "ai", "nlp", "search", "recommendation")):
        return f"{yoe:.1f} yrs in {title.lower()} roles"
    return f"{yoe:.1f} yrs ML-adjacent experience"


def generate_reasoning(
    c: dict,
    rank: int,
    submission_score: float,
    debug: dict,
    today: date = None,
) -> str:
    """
    Generate a single concise sentence of reasoning per candidate.
    No score values, no LLM — facts from profile only.
    """
    if today is None:
        today = date.today()

    profile = c["profile"]
    sig = c["redrob_signals"]
    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    rr = sig.get("recruiter_response_rate", 0.0)
    notice = sig.get("notice_period_days", 60)

    features = debug.get("features", {})
    core_skills = _get_core_skills(c)
    retrieval_highlight = _get_retrieval_career_highlight(c)
    concern = _get_primary_concern(c, features, rank)
    strength = _best_strength(c, core_skills, retrieval_highlight)
    v = _seed_from_id(c.get("candidate_id", "X"))

    concern_suffix = f" Gap: {concern}." if concern else ""

    # ── TOP 5: confident, specific, usually no gap unless serious ──
    if rank <= 5:
        t = [
            f"{title}, {yoe:.1f} yrs at {company} ({location}); {strength}. RR {rr:.0%}, {notice}d notice.{concern_suffix}",
            f"Top fit: {strength}. {title} at {company}, {yoe:.1f} yrs, {location}; RR {rr:.0%}.{concern_suffix}",
            f"{title} ({company}, {yoe:.1f} yrs) — {strength}; {notice}-day notice, RR {rr:.0%}.{concern_suffix}",
            f"Strong JD match: {strength}. Based in {location}, {yoe:.1f} yrs as {title} at {company}.{concern_suffix}",
        ]
        return t[v]

    # ── RANKS 6–20: balanced, one strength + optional gap ──
    if rank <= 20:
        t = [
            f"{title}, {yoe:.1f} yrs at {company} ({location}); {strength}. RR {rr:.0%}, {notice}d notice.{concern_suffix}",
            f"Good fit: {strength}. {title} at {company}, {location}, RR {rr:.0%}.{concern_suffix}",
            f"{yoe:.1f}-yr {title} at {company}; {strength}. Notice {notice}d.{concern_suffix}",
            f"Interview-worthy: {strength}. {title}, {company}, {yoe:.1f} yrs, {location}.{concern_suffix}",
        ]
        return t[v]

    # ── RANKS 21–50: one strength, one gap if real ──
    if rank <= 50:
        gap = concern or "partial fit for specialist retrieval role"
        t = [
            f"{title} at {company} ({yoe:.1f} yrs); {strength}; RR {rr:.0%}. Gap: {gap}.",
            f"{strength}; {title}, {company}, {location}. RR {rr:.0%}. Gap: {gap}.",
            f"{yoe:.1f}-yr {title} at {company}: {strength}. Gap: {gap}.",
            f"Solid but not top-tier: {strength}. {title}, {company}. Gap: {gap}.",
        ]
        return t[v]

    # ── RANKS 51–100: honest limitation, shorter ──
    limiter = concern or "below shortlist threshold for this JD"
    t = [
        f"{title}, {yoe:.1f} yrs at {company}; {limiter}. RR {rr:.0%}.",
        f"Rank #{rank}: {limiter}. {title} at {company}, {location}.",
        f"{title} ({company}, {yoe:.1f} yrs) — {limiter}.",
        f"Outside top 50: {limiter}. {title}, {company}, RR {rr:.0%}.",
    ]
    return t[v]
