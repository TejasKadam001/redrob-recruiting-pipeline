"""
features.py — Complete feature engineering pipeline.
Extracts ~40 engineered features per candidate. Every function
is self-contained, testable, and documented.
"""

import math
from datetime import date
from typing import Dict, Any

from src.config import (
    TITLE_TIER_1, TITLE_TIER_2, TITLE_TIER_3, TITLE_IRRELEVANT,
    SKILL_ONTOLOGY, CONSULTING_GIANTS, RETRIEVAL_SPECIFIC_TERMS,
    PREFERRED_CITIES, ACCEPTABLE_CITIES, WRONG_DOMAIN_SIGNALS,
    WRONG_DOMAIN_MITIGATING, FEATURE_WEIGHTS,
)


# ─────────────────────────────────────────────────────────────
# HELPER: safe date parsing
# ─────────────────────────────────────────────────────────────

def _safe_days_since(date_str: str, today: date) -> int:
    try:
        return (today - date.fromisoformat(date_str)).days
    except Exception:
        return 365  # default: 1 year ago if parsing fails


# ─────────────────────────────────────────────────────────────
# GROUP A — DOMAIN FIT  (35%)
# ─────────────────────────────────────────────────────────────

def feat_title_tier(c: dict) -> float:
    """Title relevance tier: 1.0 = ML/AI, 0.65 = adjacent, 0.05 = irrelevant."""
    title = c["profile"].get("current_title", "").lower()
    if any(t in title for t in TITLE_TIER_1):
        return 1.0
    if any(t in title for t in TITLE_TIER_2):
        return 0.65
    if any(t in title for t in TITLE_TIER_3):
        return 0.35
    if any(t in title for t in TITLE_IRRELEVANT):
        return 0.05
    return 0.30  # Unknown


def feat_career_trajectory_ml(c: dict) -> float:
    """Are the last 3 roles moving toward ML/AI? Rewards career pivots."""
    recent_roles = sorted(
        c.get("career_history", []),
        key=lambda r: r.get("end_date") or "9999",
        reverse=True
    )[:3]
    if not recent_roles:
        return 0.0
    ml_terms = {
        "machine learning", "deep learning", "data scientist",
        "nlp", "natural language", "search", "retrieval", "ranking",
        "recommendation", "applied scientist", "ai engineer",
        "ml engineer", "research engineer", "computer vision",
    }
    hits = sum(
        1 for r in recent_roles
        if any(t in r.get("title", "").lower() for t in ml_terms)
    )
    return hits / len(recent_roles)


def feat_product_company_ratio(c: dict) -> float:
    """Fraction of career spent at product companies (not consulting giants)."""
    career = c.get("career_history", [])
    if not career:
        return 0.5
    consulting_count = sum(
        1 for r in career
        if any(g in r.get("company", "").lower() for g in CONSULTING_GIANTS)
    )
    return 1.0 - (consulting_count / len(career))


def feat_retrieval_specificity(c: dict) -> float:
    """
    Did they actually do RETRIEVAL/RANKING work — not just generic ML?
    This is the #1 differentiator for this JD.
    """
    text = " ".join([
        c["profile"].get("summary", ""),
        " ".join(r.get("description", "") + " " + r.get("title", "")
                 for r in c.get("career_history", [])),
    ]).lower()

    hits = sum(1 for term in RETRIEVAL_SPECIFIC_TERMS if term in text)

    # Graduated scoring
    if hits >= 6:
        return 1.0
    if hits >= 4:
        return 0.85
    if hits >= 2:
        return 0.65
    if hits >= 1:
        return 0.45
    # Check for generic ML at least
    generic_ml = {"machine learning", "deep learning", "neural network",
                  "classification", "regression", "model training"}
    generic_hits = sum(1 for t in generic_ml if t in text)
    return 0.30 if generic_hits >= 2 else 0.15


def feat_industry_relevance(c: dict) -> float:
    """Industry of current company."""
    industry = c["profile"].get("current_industry", "").lower()
    tech_industries = {
        "e-commerce", "fintech", "saas", "edtech", "healthtech",
        "technology", "software", "internet", "ai", "ml", "data",
        "retail tech", "marketplace", "logistics tech",
    }
    if any(t in industry for t in tech_industries):
        return 1.0
    if "it services" in industry or "information technology" in industry:
        return 0.55
    if "banking" in industry or "finance" in industry or "consulting" in industry:
        return 0.40
    return 0.30


# ─────────────────────────────────────────────────────────────
# GROUP B — SKILLS INTELLIGENCE  (25%)
# ─────────────────────────────────────────────────────────────

def feat_weighted_skill_score(c: dict) -> float:
    """
    Skills score weighted by:
    - Ontology tier (CRITICAL > IMPORTANT > NICE_TO_HAVE)
    - Endorsements (community trust)
    - Proficiency level
    - Duration (sanity check)
    """
    tier_weight = {"CRITICAL": 1.0, "IMPORTANT": 0.70, "NICE_TO_HAVE": 0.35}
    prof_weight = {"expert": 1.0, "advanced": 0.80, "intermediate": 0.50, "beginner": 0.20}

    all_ontology_matches = {}
    for cat, data in SKILL_ONTOLOGY.items():
        for m in data["matches"]:
            all_ontology_matches[m] = (cat, tier_weight[data["tier"]])

    total_score = 0.0
    for skill in c.get("skills", []):
        sname = skill.get("name", "").lower()
        matched_tier_w = 0.0

        for match_key, (cat, tw) in all_ontology_matches.items():
            if match_key in sname or sname in match_key:
                matched_tier_w = max(matched_tier_w, tw)

        if matched_tier_w == 0.0:
            continue

        endorsements = skill.get("endorsements", 0)
        trust = min(1.0, 0.5 + endorsements / 30.0)  # 0.5 base, 1.0 at 15 endorsements

        prof = prof_weight.get(skill.get("proficiency", "intermediate"), 0.50)

        duration = skill.get("duration_months", 0)
        dur_trust = min(1.0, duration / 12.0) if duration > 0 else 0.30

        total_score += matched_tier_w * trust * prof * dur_trust

    return min(1.0, total_score / 5.0)  # Normalise: 5 strong skills = 1.0


def feat_skill_depth(c: dict) -> float:
    """What fraction of their skills are advanced or expert?"""
    skills = c.get("skills", [])
    if not skills:
        return 0.0
    deep = sum(1 for s in skills if s.get("proficiency") in ("advanced", "expert"))
    return min(1.0, deep / 6.0)  # 6 deep skills = 1.0


def feat_avg_assessment_score(c: dict) -> float:
    """Average Redrob platform assessment score (verified, not self-reported)."""
    scores = c["redrob_signals"].get("skill_assessment_scores", {})
    if not scores:
        return 0.50  # Neutral — not assessed yet
    return sum(scores.values()) / len(scores) / 100.0


def feat_num_assessments(c: dict) -> float:
    """Number of platform assessments completed (shows engagement)."""
    n = len(c["redrob_signals"].get("skill_assessment_scores", {}))
    return min(1.0, n / 5.0)


def feat_github_activity(c: dict) -> float:
    """
    GitHub activity score. -1 = no GitHub linked.
    This JD says 'this role writes code' — GitHub matters a lot.
    """
    score = c["redrob_signals"].get("github_activity_score", -1)
    if score == -1:
        return 0.15  # No GitHub is a mild negative for ML engineer
    return score / 100.0


def feat_community_trust(c: dict) -> float:
    """Total endorsements received on platform."""
    n = c["redrob_signals"].get("endorsements_received", 0)
    return min(1.0, n / 80.0)


# ─────────────────────────────────────────────────────────────
# GROUP C — EXPERIENCE QUALITY  (20%)
# ─────────────────────────────────────────────────────────────

def feat_yoe_fit(c: dict) -> float:
    """
    Years of experience vs JD range (5–9 yrs preferred).
    Not a hard cutoff — graduated scoring.
    """
    yoe = c["profile"].get("years_of_experience", 0)
    if 5 <= yoe <= 9:
        return 1.0
    if 4 <= yoe < 5:
        return 0.85
    if 9 < yoe <= 12:
        return 0.80
    if 3 <= yoe < 4:
        return 0.55
    if 12 < yoe <= 15:
        return 0.65
    if yoe < 3:
        return 0.25
    return 0.40  # 15+ years: over-qualified signal


def feat_recent_ml_work(c: dict) -> float:
    """Is their most recent job ML/AI/Search related?"""
    career = sorted(
        c.get("career_history", []),
        key=lambda r: r.get("end_date") or "9999",
        reverse=True
    )
    if not career:
        return 0.0
    most_recent_title = career[0].get("title", "").lower()
    ml_terms = {"ml", "machine learning", "ai", "data scientist", "nlp",
                "search", "recommendation", "retrieval", "applied scientist"}
    return 1.0 if any(t in most_recent_title for t in ml_terms) else 0.25


def feat_longest_ml_role_months(c: dict) -> float:
    """Longest tenure in an ML-adjacent role (shows sustained commitment)."""
    ml_durations = [
        r.get("duration_months", 0)
        for r in c.get("career_history", [])
        if any(t in r.get("title", "").lower()
               for t in ["ml", "ai", "data", "nlp", "search", "recommendation"])
    ]
    if not ml_durations:
        return 0.0
    return min(1.0, max(ml_durations) / 36.0)  # 3 year longest = 1.0


def feat_career_stability(c: dict) -> float:
    """
    Average tenure across roles. JD says they want someone for 3+ years.
    Job-hopping (< 12 month average) is a mild negative.
    """
    career = c.get("career_history", [])
    if len(career) < 2:
        return 0.60
    avg_months = sum(r.get("duration_months", 0) for r in career) / len(career)
    return min(1.0, avg_months / 24.0)  # 2-year average = 1.0


def feat_education_tier(c: dict) -> float:
    """Best education institution tier."""
    edu = c.get("education", [])
    if not edu:
        return 0.30
    tier_map = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.50, "tier_4": 0.30, "unknown": 0.30}
    return max(tier_map.get(e.get("tier", "unknown"), 0.30) for e in edu)


# ─────────────────────────────────────────────────────────────
# GROUP D — BEHAVIORAL AVAILABILITY  (15%)
# ─────────────────────────────────────────────────────────────

def feat_recency_score(c: dict, today: date = None) -> float:
    """Exponential decay: active today = 1.0, active 6 months ago ≈ 0.05."""
    if today is None:
        today = date.today()
    last_active = c["redrob_signals"].get("last_active_date", "2020-01-01")
    days = _safe_days_since(last_active, today)
    # Half-life of 45 days — strong recency signal
    return math.exp(-days / 45.0)


def feat_response_rate(c: dict) -> float:
    """Recruiter response rate [0, 1]. Direct proxy for reachability."""
    return c["redrob_signals"].get("recruiter_response_rate", 0.30)


def feat_open_to_work(c: dict) -> float:
    """Is the candidate explicitly open to work?"""
    return 1.0 if c["redrob_signals"].get("open_to_work_flag", False) else 0.25


def feat_notice_period_score(c: dict) -> float:
    """
    JD: 'We'd love sub-30-day notice. We can buy out up to 30 days.'
    Explicit scoring based on JD language.
    """
    days = c["redrob_signals"].get("notice_period_days", 60)
    if days <= 15:   return 1.00   # Immediate / can start now
    if days <= 30:   return 0.90   # JD's sweet spot — buyout possible
    if days <= 60:   return 0.65   # Manageable
    if days <= 90:   return 0.40   # Getting long
    return 0.15                     # 90+ days — challenging


def feat_interview_completion(c: dict) -> float:
    """Do they show up for interviews they commit to?"""
    return c["redrob_signals"].get("interview_completion_rate", 0.50)


def feat_profile_trust(c: dict) -> float:
    """Verified contact info + LinkedIn = trusted profile."""
    sig = c["redrob_signals"]
    score = (
        (0.40 if sig.get("verified_email", False) else 0.0) +
        (0.35 if sig.get("verified_phone", False) else 0.0) +
        (0.25 if sig.get("linkedin_connected", False) else 0.0)
    )
    return score


def feat_profile_completeness(c: dict) -> float:
    """Platform profile completeness percentage."""
    return c["redrob_signals"].get("profile_completeness_score", 50) / 100.0


# ─────────────────────────────────────────────────────────────
# GROUP E — LOGISTICS  (5%)
# ─────────────────────────────────────────────────────────────

def feat_location_fit(c: dict) -> float:
    """JD prefers Noida/Pune. Other Indian cities OK. Abroad needs relocation."""
    loc = c["profile"].get("location", "").lower()
    country = c["profile"].get("country", "").lower()
    sig = c["redrob_signals"]

    if any(city in loc for city in PREFERRED_CITIES):
        return 1.0
    if any(city in loc for city in ACCEPTABLE_CITIES):
        return 0.75
    if country == "india":
        return 0.55
    if sig.get("willing_to_relocate", False):
        return 0.40
    return 0.15  # Abroad, not willing to relocate


def feat_salary_fit(c: dict) -> float:
    """
    JD is a startup senior role — approx 25–70 LPA range is reasonable.
    Too high = probably won't accept. Too low = might be junior.
    """
    salary = c["redrob_signals"].get("expected_salary_range_inr_lpa", {})
    sal_min = salary.get("min", 20)
    sal_max = salary.get("max", 40)

    if 18 <= sal_min <= 70 and 25 <= sal_max <= 100:
        return 1.0   # Reasonable senior range
    if sal_min > 120:
        return 0.25  # Likely too expensive for startup
    if sal_max < 8:
        return 0.40  # Likely too junior
    return 0.65


# ─────────────────────────────────────────────────────────────
# PENALTY FEATURES — Applied multiplicatively
# ─────────────────────────────────────────────────────────────

def penalty_keyword_stuffer(c: dict) -> float:
    """
    Penalty 0.0–1.0 for keyword stuffers:
    irrelevant title + tons of AI skills + low endorsements + no ML in career.
    """
    title = c["profile"].get("current_title", "").lower()
    is_irrelevant_title = any(t in title for t in TITLE_IRRELEVANT)
    if not is_irrelevant_title:
        return 0.0

    skills = c.get("skills", [])
    all_match_keys = {m for cat in SKILL_ONTOLOGY.values() for m in cat["matches"]}
    ai_skills = [
        s for s in skills
        if any(m in s.get("name", "").lower() for m in all_match_keys)
    ]
    if len(ai_skills) < 4:
        return 0.0  # Not enough AI skills to flag

    avg_endorsements = sum(s.get("endorsements", 0) for s in ai_skills) / len(ai_skills)

    career_text = " ".join(r.get("description", "") for r in c.get("career_history", [])).lower()
    ml_in_career = sum(1 for kw in {"machine learning", "retrieval", "nlp", "recommendation",
                                     "ranking", "embedding", "neural", "ai model"}
                       if kw in career_text)

    if len(ai_skills) >= 5 and avg_endorsements < 4 and ml_in_career < 2:
        return min(0.50, 0.10 * len(ai_skills))
    return 0.0


def penalty_consulting_only(c: dict) -> float:
    """Penalty for pure consulting background (JD explicitly mentions this)."""
    career = c.get("career_history", [])
    if not career:
        return 0.0
    consulting_count = sum(
        1 for r in career
        if any(g in r.get("company", "").lower() for g in CONSULTING_GIANTS)
    )
    ratio = consulting_count / len(career)
    if ratio >= 1.0:
        return 0.35   # Full consulting career
    if ratio >= 0.75:
        return 0.15   # Mostly consulting
    return 0.0


def penalty_wrong_domain(c: dict) -> float:
    """Penalty for CV/Speech/Robotics primary without NLP/IR exposure."""
    all_text = " ".join([
        c["profile"].get("summary", ""),
        " ".join(r.get("description", "") + " " + r.get("title", "")
                 for r in c.get("career_history", [])),
    ]).lower()

    wrong_hits = sum(1 for t in WRONG_DOMAIN_SIGNALS if t in all_text)
    mitigating_hits = sum(1 for t in WRONG_DOMAIN_MITIGATING if t in all_text)

    if wrong_hits >= 3 and mitigating_hits < 2:
        return 0.20  # Clearly wrong domain
    return 0.0


def boost_hidden_gem(c: dict) -> float:
    """
    Boost for candidates who describe retrieval work in plain language.
    These are the diamonds other teams miss.
    """
    text = " ".join([
        c["profile"].get("summary", ""),
        " ".join(r.get("description", "") for r in c.get("career_history", [])),
    ]).lower()

    gems = {
        "recommendation engine": 0.18, "search system": 0.16,
        "document ranking": 0.18, "relevance scoring": 0.16,
        "personalization engine": 0.15, "similarity matching": 0.16,
        "candidate matching": 0.18, "ranking algorithm": 0.18,
        "search relevance": 0.18, "feed algorithm": 0.14,
        "job matching": 0.18, "semantic matching": 0.18,
        "nearest neighbor": 0.16, "inverted index": 0.16,
        "production system": 0.12, "shipped to production": 0.12,
        "serving millions": 0.14, "real users": 0.10,
    }
    boost = 0.0
    for term, weight in gems.items():
        if term in text:
            boost = max(boost, weight)  # take the best match

    return boost


# ─────────────────────────────────────────────────────────────
# GROUP F — ADVANCED SIGNALS (New — unlocks top-5% ranking)
# ─────────────────────────────────────────────────────────────

def feat_skill_pair_synergy(c: dict) -> float:
    """
    Skill COMBINATIONS are more valuable than individual skills.
    Having embeddings + vector_search together = retrieval engineer.
    Having ranking + evaluation together = production ML engineer.
    Most teams miss this — they score skills independently.
    """
    all_skill_text = " ".join(
        s.get("name", "").lower() for s in c.get("skills", [])
    )
    all_text = all_skill_text + " " + " ".join(
        r.get("description", "").lower() for r in c.get("career_history", [])
    )

    # Power pairs — each pair worth a boost
    POWER_PAIRS = [
        ({"faiss", "annoy", "hnswlib", "scann", "vector search", "ann"},
         {"sentence-transformer", "bge", "bert", "embedding", "e5"},
         0.25),  # Vector search + embeddings = retrieval engineer
        ({"ranking", "rerank", "cross-encoder", "ltr", "learning to rank"},
         {"ndcg", "mrr", "a/b", "evaluation", "offline eval"},
         0.22),  # Ranking + evaluation = rigorous ML engineer
        ({"elasticsearch", "opensearch", "solr", "lucene"},
         {"embedding", "vector", "semantic", "dense"},
         0.20),  # Hybrid search expertise
        ({"pytorch", "tensorflow"},
         {"production", "serving", "deploy", "inference", "api"},
         0.15),  # Trains + ships models
        ({"nlp", "natural language", "text"},
         {"retrieval", "search", "ranking", "recommendation"},
         0.18),  # NLP + IR = perfect combo
    ]

    best_synergy = 0.0
    for group_a, group_b, boost in POWER_PAIRS:
        has_a = any(t in all_text for t in group_a)
        has_b = any(t in all_text for t in group_b)
        if has_a and has_b:
            best_synergy = max(best_synergy, boost)

    return best_synergy


def feat_quantified_impact(c: dict) -> float:
    """
    Real engineers measure their work. Candidates who write '10M users',
    '40% latency reduction', 'team of 8' are describing real production work.
    This distinguishes doers from resume-padders.
    """
    import re
    all_text = " ".join(
        r.get("description", "") for r in c.get("career_history", [])
    )
    if not all_text:
        return 0.0

    # Pattern: number + scale/unit near an impact word
    # e.g. "10M users", "40% reduction", "500K requests/sec", "team of 12"
    scale_hits = len(re.findall(
        r'\b\d+\.?\d*\s*[MKBmkb]?\s*(?:users?|requests?|queries|'
        r'candidates?|documents?|records?|items?|transactions?)\b',
        all_text, re.IGNORECASE
    ))
    pct_hits = len(re.findall(
        r'\b\d+\.?\d*\s*%\s*(?:reduction|improvement|increase|faster|'
        r'accuracy|precision|recall|latency|throughput)\b',
        all_text, re.IGNORECASE
    ))
    team_hits = len(re.findall(
        r'\bteam\s+of\s+\d+|\bmentored?\s+\d+|\bled\s+\d+\s+engineer',
        all_text, re.IGNORECASE
    ))

    total = scale_hits + pct_hits + team_hits
    if total >= 5: return 1.0
    if total >= 3: return 0.75
    if total >= 1: return 0.45
    return 0.0


def feat_career_progression_score(c: dict) -> float:
    """
    Is their seniority level consistent with their YoE?
    And did they PROGRESS (Junior → Mid → Senior) or stagnate?
    Inflation: 'Senior ML' after 1.5 yrs = red flag.
    Legitimate: 'Staff Engineer' after 8 yrs at Google = credible.
    """
    yoe = c["profile"].get("years_of_experience", 0)
    title = c["profile"].get("current_title", "").lower()

    senior_terms = {"senior", "staff", "lead", "principal", "head", "director"}
    junior_terms = {"junior", "associate", "intern", "trainee", "entry"}

    is_senior = any(t in title for t in senior_terms)
    is_junior = any(t in title for t in junior_terms)

    # YoE vs seniority alignment
    if is_junior and yoe >= 6:   return 0.55  # Over-experienced for junior title
    if is_senior and yoe < 2:    return 0.35  # Inflated senior claim
    if is_senior and yoe >= 5:   return 1.0   # Consistent senior
    if not is_senior and not is_junior:
        if 3 <= yoe <= 8:        return 0.80  # Mid-level with appropriate YoE
        if yoe >= 8:             return 0.70  # Experienced, just hasn't leveled up

    return 0.65  # Default reasonable


def feat_skill_recency(c: dict) -> float:
    """
    A skill used in a job 3 months ago > same skill from 5 years ago.
    Weight skills by how recently they appeared in career descriptions.
    """
    career = sorted(
        c.get("career_history", []),
        key=lambda r: r.get("end_date") or "9999",
        reverse=True  # Most recent first
    )
    if not career:
        return 0.0

    all_matches = {m for cat in SKILL_ONTOLOGY.values() for m in cat["matches"]}
    recency_score = 0.0
    decay = 1.0  # Most recent job = 1.0, second = 0.6, third = 0.36

    for role in career[:4]:
        desc = role.get("description", "").lower() + " " + role.get("title", "").lower()
        relevant_hits = sum(1 for m in all_matches if m in desc)
        recency_score += relevant_hits * decay
        decay *= 0.60  # Each older job contributes 60% less

    return min(1.0, recency_score / 6.0)  # Normalize: 6 recency-weighted hits = 1.0


def feat_degree_field(c: dict) -> float:
    """
    Degree field relevance. CS/AI/ML/Stats degree > other engineering > unrelated.
    For ML roles, degree field is a meaningful signal for deep understanding.
    """
    edu = c.get("education", [])
    if not edu:
        return 0.30

    best = 0.30
    for e in edu:
        field = e.get("field_of_study", "").lower()
        degree = e.get("degree", "").lower()

        if any(t in field for t in ["machine learning", "artificial intelligence",
                                     "data science", "statistics", "math"]):
            best = max(best, 1.0)
        elif any(t in field for t in ["computer science", "computer engineering",
                                       "software engineering", "information technology"]):
            best = max(best, 0.85)
        elif any(t in field for t in ["electronics", "electrical", "mechanical",
                                       "instrumentation", "physics"]):
            best = max(best, 0.55)
        elif "phd" in degree or "ph.d" in degree:
            best = max(best, 0.80)  # PhD in anything = serious academic training
        else:
            best = max(best, 0.30)

    return best


def feat_applications_submitted(c: dict) -> float:
    """
    Actively applying to jobs = high conversion probability.
    0 apps = passive. 5+ apps = hunting. Recruiters love active seekers.
    This signal exists in the schema but was never used before.
    """
    apps = c["redrob_signals"].get("applications_submitted_30d", 0)
    if apps >= 5:  return 1.0
    if apps >= 2:  return 0.75
    if apps >= 1:  return 0.50
    return 0.20   # Passive candidate


def feat_offer_acceptance_rate(c: dict) -> float:
    """
    Will they actually JOIN after an offer? Critical for time-to-hire.
    -1 = never received an offer (new to market or very passive).
    0.8 = accepts 80% of offers (decisive, good for fast-moving startups).
    """
    oar = c["redrob_signals"].get("offer_acceptance_rate", -1)
    if oar == -1:  return 0.40   # Unknown — neutral
    if oar >= 0.7: return 1.0
    if oar >= 0.4: return 0.65
    return 0.30   # Low acceptance — hard to close


def feat_platform_seniority(c: dict) -> float:
    """
    How long have they been on Redrob? Long-standing active members
    have richer, more reliable signal than brand-new profiles.
    Also partial proxy for how serious they are about job searching.
    """
    signup = c["redrob_signals"].get("signup_date", "2022-01-01")
    try:
        days_on_platform = (date.today() - date.fromisoformat(signup)).days
    except Exception:
        days_on_platform = 0
    return min(1.0, days_on_platform / 730.0)  # 2 years = full score


# ─────────────────────────────────────────────────────────────
# MASTER FEATURE VECTOR
# ─────────────────────────────────────────────────────────────

def build_feature_vector(c: dict, today: date = None) -> dict:
    """Build all features for a single candidate. Returns dict of feature_name → value."""
    if today is None:
        today = date.today()

    return {
        # Group A: Domain Fit
        "title_tier":               feat_title_tier(c),
        "career_trajectory_ml":     feat_career_trajectory_ml(c),
        "product_company_ratio":    feat_product_company_ratio(c),
        "retrieval_specificity":    feat_retrieval_specificity(c),
        "industry_relevance":       feat_industry_relevance(c),
        # Group B: Skills
        "weighted_skill_score":     feat_weighted_skill_score(c),
        "skill_depth":              feat_skill_depth(c),
        "avg_assessment_score":     feat_avg_assessment_score(c),
        "num_assessments":          feat_num_assessments(c),
        "github_activity":          feat_github_activity(c),
        "community_trust":          feat_community_trust(c),
        # Group C: Experience
        "yoe_fit":                  feat_yoe_fit(c),
        "recent_ml_work":           feat_recent_ml_work(c),
        "longest_ml_role_months":   feat_longest_ml_role_months(c),
        "career_stability":         feat_career_stability(c),
        "education_tier":           feat_education_tier(c),
        # Group D: Behavioral
        "recency_score":            feat_recency_score(c, today),
        "response_rate":            feat_response_rate(c),
        "open_to_work":             feat_open_to_work(c),
        "notice_period_score":      feat_notice_period_score(c),
        "interview_completion":     feat_interview_completion(c),
        "profile_trust":            feat_profile_trust(c),
        "profile_completeness":     feat_profile_completeness(c),
        # Group E: Logistics
        "location_fit":             feat_location_fit(c),
        "salary_fit":               feat_salary_fit(c),
        # Group F: Advanced signals
        "skill_pair_synergy":         feat_skill_pair_synergy(c),
        "quantified_impact":          feat_quantified_impact(c),
        "career_progression_score":   feat_career_progression_score(c),
        "skill_recency":              feat_skill_recency(c),
        "degree_field":               feat_degree_field(c),
        # Group G: Conversion signals (new — uses previously ignored schema fields)
        "applications_submitted":     feat_applications_submitted(c),
        "offer_acceptance_rate":      feat_offer_acceptance_rate(c),
        "platform_seniority":         feat_platform_seniority(c),
    }


def compute_rule_score(features: dict) -> float:
    """Weighted sum of features → single rule-based score."""
    return sum(features.get(k, 0.0) * w for k, w in FEATURE_WEIGHTS.items())


def compute_behavioral_multiplier(c: dict, today: date = None) -> float:
    """
    Calibrated Behavioral Penalty (Bayesian Update)
    Treats response_rate and recency as a Bayesian prior for the 
    probability of candidate engagement.
    """
    if today is None:
        today = date.today()
    sig = c["redrob_signals"]

    # 1. Base components
    last_active = sig.get("last_active_date", "2020-01-01")
    days = _safe_days_since(last_active, today)
    recency_weight = math.exp(-days / 90.0)  # Half-life 90 days

    rr = sig.get("recruiter_response_rate", 0.30)
    otw = 1.0 if sig.get("open_to_work_flag", False) else 0.0

    # 2. Bayesian Update using Beta Distribution
    # Platform prior: alpha=2 (responses), beta=8 (ignores) => mean 20%
    prior_alpha = 2.0
    prior_beta  = 8.0

    # Evidence: treat `rr` as observed success rate over `N` recent opportunities
    # N is high if recently active and OTW, low if ghosted
    evidence_N = 20.0 * recency_weight + (10.0 if otw else 0)
    
    observed_successes = evidence_N * rr
    observed_failures  = evidence_N * (1.0 - rr)

    posterior_alpha = prior_alpha + observed_successes
    posterior_beta  = prior_beta + observed_failures

    expected_engagement_prob = posterior_alpha / (posterior_alpha + posterior_beta)

    # 3. Add deterministic logistics boosts
    notice = feat_notice_period_score(c)
    icr = sig.get("interview_completion_rate", 0.50)
    github = max(0, sig.get("github_activity_score", 0)) / 100.0
    
    logistics_score = (0.5 * notice) + (0.3 * icr) + (0.2 * github)

    # Final multiplier combines Bayesian expected engagement with logistics.
    # Scaled to maintain the 0.35 - 1.25 bounds for the overall ranker architecture.
    final_mult = 0.35 + (expected_engagement_prob * 0.70) + (logistics_score * 0.20)
    
    return max(0.35, min(1.25, final_mult))
