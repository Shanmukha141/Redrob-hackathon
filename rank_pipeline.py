import zipfile
import os
import json
import pprint
from datetime import datetime
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity




# file extraction

candidates = []

path = "candidates.jsonl"

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if line:
            candidates.append(json.loads(line))

print("Total candidates:", len(candidates))
#============================================================
# DATA CLEANING
#============================================================
# Initialize pretty printer for clean output
pp = pprint.PrettyPrinter(indent=2, width=80)

def clean_candidates_data(candidates_list):
    viable_candidates = []
    discarded_count = 0

    # Pre-defined traps and keywords based on the Job Description
    consulting_firms = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'}
    irrelevant_titles = [
        'marketing manager', 'accountant', 'hr manager',
        'operations manager', 'customer support', 'civil engineer', 'sales'
    ]
    research_keywords = ['research assistant', 'postdoc', 'phd researcher', 'academic', 'lab']
    production_keywords = ['deployed', 'production', 'users', 'scale', 'pipeline', 'infrastructure', 'shipped']

    # Core AI/ML target domain keywords
    ai_ml_keywords = {
        'ml', 'machine learning', 'ai', 'artificial intelligence', 'nlp',
        'computer vision', 'llm', 'deep learning', 'neural networks',
        'pytorch', 'tensorflow', 'embeddings', 'retrieval', 'ranking',
        'generative ai', 'gans', 'pinecone', 'qdrant', 'weaviate', 'milvus'
    }

    for candidate in candidates_list:
        profile = candidate.get('profile', {})
        signals = candidate.get('redrob_signals', {})
        history = candidate.get('career_history', [])
        skills = candidate.get('skills', [])

        is_viable = True

        # 1. Behavioral Trap Filter
        if signals.get('recruiter_response_rate', 1.0) < 0.10:
            is_viable = False

        # 2. Irrelevant Title Filter (Keyword Stuffers)
        current_title = profile.get('current_title', '').lower()
        if any(trap in current_title for trap in irrelevant_titles):
            is_viable = False

        # 3. "Consulting Only" Filter
        worked_at_product_company = False
        for role in history:
            if role.get('company', '').lower() not in consulting_firms:
                worked_at_product_company = True
                break
        if not worked_at_product_company and len(history) > 0:
            is_viable = False

        # 4. Job Hopping / Title Chasing Filter
        if len(history) >= 3:
            total_months = sum(role.get('duration_months', 0) for role in history)
            avg_duration = total_months / len(history)
            if avg_duration <= 18:
                is_viable = False

        # 5. Pure Research vs Production Filter
        production_signals = 0
        research_signals = 0
        for role in history:
            title = role.get('title', '').lower()
            desc = role.get('description', '').lower()

            if any(k in title for k in research_keywords):
                research_signals += 1
            if any(k in desc for k in production_keywords):
                production_signals += 1

        if research_signals > 0 and production_signals == 0:
            is_viable = False

        # ==========================================================
        # NEW CRITICAL FILTERS: TARGETED YEARS OF EXPERIENCE (YoE)
        # ==========================================================

        # 6. Minimum Overall Experience (5+ Years)
        overall_yoe = profile.get('years_of_experience') or 0.0
        if overall_yoe < 5.0:
            is_viable = False
        # ==========================================================
        # 7. Location / Relocation Filter
        # ==========================================================

        location = (
            profile.get('location') or ''
        ).lower()

        willing_to_relocate = signals.get(
            'willing_to_relocate',
            False
        )

        preferred_locations = {

            'pune',

            'noida',

            'hyderabad',

            'mumbai',

            'delhi ncr ',
            'delhi'

        }

        location_match = any(
            city in location
            for city in preferred_locations
        )

        if not (
            location_match
            or
            willing_to_relocate
        ):
            is_viable = False

        # ==========================================================
        # 8. Notice Period Filter
        # ==========================================================

        notice_period = signals.get(
            'notice_period_days',
            90
        )

        # Reject candidates with very long notice periods
        if notice_period > 90:
            is_viable = False

        # ==========================================================



        # 9. Minimum Core AI/ML Experience (4+ Years / 48 Months)
        # Prevents general high-YoE profiles without deep AI/ML track records

        total_ai_months = 0
        for skill in skills:
            skill_name = (skill.get('name') or '').lower()
            if any(kw in skill_name for kw in ai_ml_keywords):
                total_ai_months += (skill.get('duration_months') or 0)

        # 48 months equals 4 years of compiled project experience
        if total_ai_months < 48:
            is_viable = False

        # ==========================================================

        # Final Decision
        if is_viable:
            viable_candidates.append(candidate)
        else:
            discarded_count += 1

    return viable_candidates, discarded_count

# Run the cleaning process on your loaded 'candidates' list
print("Starting optimized data cleaning process...")
cleaned_pool, discarded = clean_candidates_data(candidates)

print("-" * 40)
print(f"Total Original Candidates: {len(candidates)}")
print(f"Candidates Discarded (Traps/Unfit): {discarded}")
print(f"Viable Candidates Remaining: {len(cleaned_pool)}")
print("-" * 40)


#==============================================================
# MODEL EVALUATION
#==============================================================


# ============================================================
# 1. INITIALIZE EMBEDDING MODEL
# ============================================================
print("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# 2. CONFIGURATION & NEW HIGH-REWARD KEYWORDS
# ============================================================
JD_TEXT = """
Deep technical depth in modern ML systems, embeddings, retrieval, ranking, recommendation systems.
LLMs, fine-tuning. Production experience with vector databases. Evaluation frameworks (NDCG, MRR, MAP, A/B Testing).
"""

EVAL_KEYWORDS = [
    "ndcg", "mrr", "map", "a/b testing", "a/b test", "offline-online",
    "offline evaluation", "learning to rank", "ltr"
]

MATCHING_KEYWORDS = [
    "candidate matching", "recruiter search", "retrieval", "ranking",
    "semantic search", "talent search", "search systems", "recommendation systems",
    "vector search", "reranking"
]

TOP_PRODUCT = {"meta", "apple", "google", "amazon", "microsoft", "razorpay", "paytm", "observe.ai", "linkedin", "uber", "netflix", "atlassian"}
SERVICE = {"tcs", "infosys", "wipro", "cognizant", "capgemini", "accenture", "mindtree", "ltimindtree", "hcl", "tech mahindra"}

# ============================================================
# 3. TEXT WEIGHTING & NULL-SAFE FUNCTIONS
# ============================================================
def build_candidate_text(candidate):
    profile = candidate.get("profile", {})
    headline = profile.get("headline") or ""
    summary = profile.get("summary") or ""
    skills = " ".join((skill.get("name") or "") for skill in candidate.get("skills", []))
    experience = " ".join((role.get("description") or "") for role in candidate.get("career_history", []))

    # Weighting logic: Headline x1, Summary x2, Skills x1, Experience x3
    weighted_text = f"{headline} " + f"{summary} " * 2 + f"{skills} " + f"{experience} " * 3
    return weighted_text

def company_score(candidate):
    score = 0
    for job in candidate.get("career_history", []):
        company = (job.get("company") or "").lower()
        if company in TOP_PRODUCT:
            score += 1.0
        elif company in SERVICE:
            score += 0.2
        else:
            score += 0.5
    return min(score / 3, 1.0)

# --- NEW TIER-BASED BEHAVIORAL SCORING ---
def behavior_score(signals):
    score = 0.0

    # 1. Open to Work
    if signals.get("open_to_work_flag") is True:
        score += 0.20

    # 2. Notice Period (Immediate joiners are gold)
    notice_days = signals.get("notice_period_days") or 90
    if notice_days <= 30:
        score += 0.20
    elif notice_days > 60:
        score -= 0.10 # Penalty for long wait

    # 3. Recruiter Response Rate (>80% gets massive boost)
    resp_rate = signals.get("recruiter_response_rate") or 0.0
    if resp_rate >= 0.80:
        score += 0.15
    elif resp_rate >= 0.50:
        score += 0.05

    # 4. Interview Completion Rate (>90% gets boost)
    int_rate = signals.get("interview_completion_rate") or 0.0
    if int_rate >= 0.90:
        score += 0.15

    # 5. GitHub Activity (Hitting that 75+ mark)
    github = signals.get("github_activity_score") or 0
    if github >= 75.0:
        score += 0.15
    elif github >= 50.0:
        score += 0.05

    # 6. Search Appearances (High market visibility)
    searches = signals.get("search_appearances") or 0
    if searches >= 500:
        score += 0.10
    elif searches >= 100:
        score += 0.05

    # 7. Saved by Recruiters (High demand indicator)
    saves = signals.get("saved_by_recruiters") or 0
    if saves >= 5:
        score += 0.15
    elif saves > 0:
        score += 0.05

    return min(score, 1.0) # Cap at a perfect 1.0 score

def get_domain_expertise_multiplier(text):
    text = text.lower()
    multiplier = 1.0

    eval_hits = sum(1 for kw in EVAL_KEYWORDS if kw in text)
    if eval_hits > 0:
        multiplier += min(eval_hits * 0.05, 0.15)

    match_hits = sum(1 for kw in MATCHING_KEYWORDS if kw in text)
    if match_hits > 0:
        multiplier += min(match_hits * 0.05, 0.20)

    return multiplier

def get_location_multiplier(candidate):
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    location = (profile.get('location') or "").lower()

    # "Will Relocate" is handled right here!
    willing_to_relocate = signals.get('willing_to_relocate') or False

    if any(city in location for city in ['pune', 'noida']):
        return 1.20
    elif willing_to_relocate:
        return 1.15
    return 0.80

# ============================================================
# 4. CORE RANKING ENGINE
# ============================================================
def rank_candidates(clean_data_pool):
    print(f"Building heavily weighted documents for {len(clean_data_pool)} candidates...")
    candidate_docs = [build_candidate_text(c) for c in clean_data_pool]

    print("Generating embeddings...")
    jd_embedding = model.encode(JD_TEXT, normalize_embeddings=True)
    candidate_embeddings = model.encode(
        candidate_docs,
        normalize_embeddings=True,
        batch_size=128,
        show_progress_bar=True
    )

    print("Computing semantic similarity and applying framework multipliers...")
    semantic_scores = cosine_similarity([jd_embedding], candidate_embeddings).flatten()

    ranked = []

    for index, candidate in enumerate(clean_data_pool):
        candidate_text = candidate_docs[index]
        signals = candidate.get("redrob_signals", {})

        sem_score = semantic_scores[index]
        beh_score = behavior_score(signals)
        comp_score = company_score(candidate)

        # Base AI Score
        base_score = (0.70 * sem_score) + (0.15 * beh_score) + (0.15 * comp_score)

        # Apply Multipliers
        loc_multiplier = get_location_multiplier(candidate)
        domain_multiplier = get_domain_expertise_multiplier(candidate_text)

        final_score = base_score * loc_multiplier * domain_multiplier

        ranked.append((candidate, final_score))

    ranked.sort(key=lambda x: (-x[1], x[0]["candidate_id"]))
    return ranked[:100]

# ============================================================
# 5. EXPORT AND REASONING GENERATOR
# ============================================================
def export_submission(ranked_candidates, output_csv="submission.csv"):
    print("Formatting output CSV...")
    rows = []

    for rank, (candidate, score) in enumerate(ranked_candidates, start=1):
        cand_id = candidate["candidate_id"]
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})

        title = profile.get("current_title") or "Engineer"
        yoe = profile.get("years_of_experience") or 0
        text = build_candidate_text(candidate).lower()

        # Dynamic Reasoning based on the new framework keywords and signals
        reasons = []
        if any(kw in text for kw in EVAL_KEYWORDS):
            reasons.append("Ranking Eval (NDCG/MRR)")
        if any(kw in text for kw in MATCHING_KEYWORDS):
            reasons.append("Search/Matching Systems")
        if (signals.get("notice_period_days") or 90) <= 30:
            reasons.append("30-Day Notice")
        if (signals.get("recruiter_response_rate") or 0) >= 0.80:
            reasons.append("Highly Responsive")

        reason_str = " | ".join(reasons) if reasons else "Strong Semantic Matrix"
        reasoning_final = f"{title} ({yoe} yrs) | {reason_str}"

        rows.append({
            "candidate_id": cand_id,
            "rank": rank,
            "score": round(score, 4),
            "reasoning": reasoning_final
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["score", "candidate_id"], ascending=[False, True])
    df["rank"] = range(1, len(df) + 1)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df)} top candidates to {output_csv}!")
    return df

# ============================================================
# EXECUTION
# ============================================================
print("Starting optimized ranking pipeline...")
final_top_100 = rank_candidates(cleaned_pool)
submission_df = export_submission(final_top_100, "shanmukha.csv")
submission_df


