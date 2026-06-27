"""
honeypot.py — Comprehensive honeypot detection with confidence scoring.
Honeypot rate > 10% in top-100 = instant disqualification.
We are aggressive but safe — use confidence scoring to avoid false positives.
"""

from datetime import date
from typing import Tuple, List
from src.config import (
    HONEYPOT_CONFIG, STRONG_HONEYPOT_FLAGS,
    MEDIUM_HONEYPOT_FLAGS, WEAK_HONEYPOT_FLAGS
)


def comprehensive_honeypot_check(c: dict) -> Tuple[bool, List[tuple]]:
    """
    Returns (is_honeypot, list_of_flags).
    Each flag is a tuple: (flag_name, detail).
    """
    flags = []
    today = date.today()
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    education = c.get("education", [])
    sigs = c.get("redrob_signals", {})
    profile = c.get("profile", {})

    # Safely compute total career months
    total_career_months = sum(max(0, r.get("duration_months", 0)) for r in career)
    claimed_yoe_months = profile.get("years_of_experience", 0) * 12

    # -------------------------------------------------------------------
    # CHECK 1: Experience vs career timeline mismatch
    # -------------------------------------------------------------------
    threshold = HONEYPOT_CONFIG["exp_mismatch_months_threshold"]
    if career and abs(total_career_months - claimed_yoe_months) > threshold:
        flags.append(("exp_mismatch", abs(total_career_months - claimed_yoe_months)))

    # -------------------------------------------------------------------
    # CHECK 2: Education end year vs career start year
    # -------------------------------------------------------------------
    if career and education:
        try:
            earliest_job_year = min(
                int(r["start_date"][:4]) for r in career if r.get("start_date")
            )
            latest_edu_end = max(
                e.get("end_year", 0) for e in education if e.get("end_year")
            )
            tolerance = HONEYPOT_CONFIG["edu_career_overlap_years"]
            if latest_edu_end > earliest_job_year + tolerance:
                flags.append(("edu_career_overlap", latest_edu_end - earliest_job_year))
        except (ValueError, TypeError):
            pass

    # -------------------------------------------------------------------
    # CHECK 3: Skill duration > total career duration
    # -------------------------------------------------------------------
    buffer = HONEYPOT_CONFIG["skill_duration_buffer_months"]
    for skill in skills:
        skill_dur = skill.get("duration_months", 0)
        if skill_dur > total_career_months + buffer and total_career_months > 0:
            flags.append(("skill_duration_impossible", skill.get("name", "?")))

    # -------------------------------------------------------------------
    # CHECK 4: Future certifications
    # -------------------------------------------------------------------
    current_year = today.year
    for cert in c.get("certifications", []):
        cert_year = cert.get("year", 0)
        if cert_year and cert_year > current_year + HONEYPOT_CONFIG["future_cert_tolerance_years"]:
            flags.append(("future_cert", cert.get("name", "?")))

    # -------------------------------------------------------------------
    # CHECK 5: Expert proficiency with tiny duration (< 6 months)
    # -------------------------------------------------------------------
    for skill in skills:
        if skill.get("proficiency") == "expert":
            dur = skill.get("duration_months", 999)
            if dur < 6:
                flags.append(("instant_expert", skill.get("name", "?")))

    # -------------------------------------------------------------------
    # CHECK 6: Signup date before platform existed (Redrob founded ~2019)
    # Old logic was inverted — it flagged everyone whose career predated signup.
    # Correct check: signup_year impossibly early (< 2018) means fake data.
    # -------------------------------------------------------------------
    signup = sigs.get("signup_date", "2020-01-01")
    try:
        signup_year = int(signup[:4])
        if signup_year < 2018:
            flags.append(("signup_before_platform_existed", 2018 - signup_year))
        # Also flag: signup date in the future
        if signup_year > date.today().year:
            flags.append(("future_signup", signup_year))
    except (ValueError, TypeError):
        pass

    # -------------------------------------------------------------------
    # CHECK 7: Salary range inverted (min > max)
    # -------------------------------------------------------------------
    salary = sigs.get("expected_salary_range_inr_lpa", {})
    sal_min = salary.get("min", 0)
    sal_max = salary.get("max", 999)
    if sal_min > sal_max and sal_min > 0:
        flags.append(("salary_range_inverted", f"{sal_min} > {sal_max}"))

    # -------------------------------------------------------------------
    # CHECK 8: Future job start dates
    # -------------------------------------------------------------------
    for r in career:
        start = r.get("start_date", "")
        if start and start > str(today):
            flags.append(("future_job", start))

    # -------------------------------------------------------------------
    # CHECK 9: Negative years of experience
    # -------------------------------------------------------------------
    if claimed_yoe_months < 0:
        flags.append(("negative_yoe", claimed_yoe_months))

    # -------------------------------------------------------------------
    # CHECK 10: Offer acceptance rate out of range (spec says -1 to 1)
    # -------------------------------------------------------------------
    oar = sigs.get("offer_acceptance_rate", 0)
    if oar not in (-1,) and not (-1 <= oar <= 1):
        flags.append(("offer_acceptance_out_of_range", oar))

    # -------------------------------------------------------------------
    # DECISION: is it a honeypot?
    # -------------------------------------------------------------------
    strong_flags = [f for f in flags if f[0] in STRONG_HONEYPOT_FLAGS]
    medium_flags = [f for f in flags if f[0] in MEDIUM_HONEYPOT_FLAGS]
    weak_flags   = [f for f in flags if f[0] in WEAK_HONEYPOT_FLAGS]

    # Any strong flag = instant honeypot
    if HONEYPOT_CONFIG["strong_flag_instant_hp"] and strong_flags:
        return True, flags

    # 2+ medium or medium+weak combination
    if len(medium_flags) >= 2:
        return True, flags
    if len(medium_flags) >= 1 and len(weak_flags) >= 2:
        return True, flags

    # 3+ weak flags
    if len(weak_flags) >= 3:
        return True, flags

    return False, flags


def honeypot_confidence(c: dict) -> float:
    """
    Returns 0.0–1.0 probability this is a honeypot.
    Used as a soft penalty instead of hard binary when confidence is low.
    """
    _, flags = comprehensive_honeypot_check(c)
    if not flags:
        return 0.0

    score = 0.0
    for flag in flags:
        if flag[0] in STRONG_HONEYPOT_FLAGS:
            score += 0.60
        elif flag[0] in MEDIUM_HONEYPOT_FLAGS:
            score += 0.30
        else:  # WEAK
            score += 0.10

    return min(1.0, score)


def apply_honeypot_penalty(raw_score: float, c: dict) -> float:
    """
    Apply honeypot penalty to a raw score.
    Definite honeypots → 0.0
    Suspicious ones → proportionally reduced
    Clean ones → unchanged
    """
    conf = honeypot_confidence(c)
    if conf >= 0.80:
        return 0.0
    elif conf >= 0.20:
        return raw_score * (1.0 - conf)
    else:
        return raw_score
