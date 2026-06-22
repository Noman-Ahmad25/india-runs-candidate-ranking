from datetime import datetime
from typing import Dict, Any

def extract_candidate_features(candidate_json: Dict[str, Any]) -> Dict[str, Any]:
    # Raw Data Hydration Block
    profile = candidate_json.get("profile", {})
    signals = candidate_json.get("redrob_signals", {})
    skills_list = candidate_json.get("skills", [])
    career_history = candidate_json.get("career_history", [])

    # ----------------------------------------------------
    # BASE METRIC NORMALIZATION & HARVESTING
    # ----------------------------------------------------
    experience_years = float(profile.get("years_of_experience", 0.0))
    response_rate = float(signals.get("recruiter_response_rate", 0.0))
    interview_completion_rate = float(signals.get("interview_completion_rate", 0.0))
    notice_period = int(signals.get("notice_period_days", 0))
    open_to_work = 1 if signals.get("open_to_work_flag") else 0
    saved_by_recruiters_30d = int(signals.get("saved_by_recruiters_30d", 0))
    search_appearance_30d = int(signals.get("search_appearance_30d", 0))

    # Clean GitHub -1.0 anomaly
    raw_github = float(signals.get("github_activity_score", -1.0))
    github_score = 0.0 if raw_github < 0 else raw_github

    # ----------------------------------------------------
    # RAW STRATEGIC TEXT CORPUS PREPARATION
    # ----------------------------------------------------
    text_blocks = [profile.get("summary", ""), profile.get("headline", "")]
    for job in career_history:
        text_blocks.append(job.get("description", ""))
    for skill in skills_list:
        text_blocks.append(skill.get("name", ""))
    full_text = " ".join([str(b).lower() for b in text_blocks if b])

    # Context Isolated Title Array Extraction
    candidate_historical_titles = [str(profile.get("current_title", "")).lower()]
    for job in career_history:
        if job.get("title"):
            candidate_historical_titles.append(str(job.get("title", "")).lower())

    # ----------------------------------------------------
    # LAYER 1: HARD FILTERS (EXPERIENCE BAND & PRODUCTION ML)
    # ----------------------------------------------------
    if 5.0 <= experience_years <= 9.0:
        experience_fit_score = 1.0
    elif 4.0 <= experience_years < 5.0:
        experience_fit_score = 0.8
    elif 9.0 < experience_years <= 12.0:
        experience_fit_score = 0.7
    elif experience_years < 4.0:
        experience_fit_score = 0.3
    else:  # > 12 Years (Slightly Weak for Senior IC scope)
        experience_fit_score = 0.5

    # Context Aware Production ML Co-occurrence Verification
    deployment_verbs = {"shipped", "deployed", "production", "launched", "live", "customer-facing", "real users", "serving"}
    ml_nouns = {"ml", "ai", "model", "features", "ranking", "retrieval", "search", "pipeline", "recommendation"}
    has_production_ml_signal = 0
    
    for job in career_history:
        job_text = f"{job.get('title', '')} {job.get('description', '')}".lower()
        if any(v in job_text for v in deployment_verbs) and any(n in job_text for n in ml_nouns):
            has_production_ml_signal = 1
            break

    # Pure Research Penalty Core Engine
    research_identifiers = {"research scientist", "phd researcher", "academic lab", "published", "paper"}
    production_neutralizers = {"production", "deployment", "users", "shipped", "live"}
    
    has_research_keywords = any(w in full_text for w in research_identifiers)
    has_production_keywords = any(w in full_text for w in production_neutralizers)
    research_only_risk = 1 if (has_research_keywords and not has_production_keywords) else 0

    # ----------------------------------------------------
    # LAYER 2 & 3: TECHNICAL DICTIONARY MATCHES & TECH SKILLS ACCOUNTING
    # ----------------------------------------------------
    RETRIEVER_KW = {"retrieval", "information retrieval", "semantic search", "vector search", "dense retrieval", "hybrid retrieval", "hybrid search", "embeddings", "sentence transformers", "bge", "e5", "rag", "bm25", "retriever"}
    VDB_KW = {"faiss", "pinecone", "qdrant", "weaviate", "milvus", "chroma", "elasticsearch", "opensearch"}
    RANKER_KW = {"ranking", "reranking", "learning to rank", "recommendation systems", "recommendation engine", "xgboost", "lightgbm", "catboost", "ltr"}
    EVAL_KW = {"ndcg", "map", "mrr", "precision@k", "recall@k", "ab testing", "a/b testing", "offline benchmark", "evaluation framework", "a/b test", "ab test", "experiment", "experimentation"}

    retrieval_keyword_hits = sum(1 for kw in RETRIEVER_KW if kw in full_text)
    vector_db_keyword_hits = sum(1 for kw in VDB_KW if kw in full_text)
    evaluation_keyword_hits = sum(1 for kw in EVAL_KW if kw in full_text)

    retrieval_skill_count = 0
    vector_db_skill_count = 0
    ranking_skill_count = 0
    advanced_skills_count = 0

    for skill in skills_list:
        name = str(skill.get("name", "")).lower()
        prof = str(skill.get("proficiency", "")).lower()
        
        if prof in ["advanced", "expert"]:
            advanced_skills_count += 1

        if any(kw in name for kw in RETRIEVER_KW) or name in VDB_KW:
            retrieval_skill_count += 1
        if name in VDB_KW:
            vector_db_skill_count += 1
        if name in RANKER_KW and prof in ["advanced", "expert"]:
            ranking_skill_count += 1

    # Layer 3: Targeted Production Retrieval Co-Occurrence Verification
    has_production_retrieval_signal = 0
    for job in career_history:
        job_text = f"{job.get('title', '')} {job.get('description', '')}".lower()
        if any(v in job_text for v in ["shipped", "deployed", "launched", "production", "live"]) and \
           any(r in job_text for r in ["retrieval", "ranking", "search", "recommendation", "embedding", "vector"]):
            has_production_retrieval_signal = 1
            break

    # ----------------------------------------------------
    # LAYER 4 & 5: ROLE FIT & PRODUCT COMPANY SIGNALS
    # ----------------------------------------------------
    CORE_TARGET_TITLES = {"ai engineer", "machine learning engineer", "ml engineer", "search engineer", "retrieval engineer", "ranking engineer", "recommendation engineer"}
    SECONDARY_TARGET_TITLES = {"backend engineer", "data engineer"}

    has_core_target_title = 1 if any(any(t in title for t in CORE_TARGET_TITLES) for title in candidate_historical_titles) else 0
    has_secondary_title = 1 if any(any(t in title for t in SECONDARY_TARGET_TITLES) for title in candidate_historical_titles) else 0
    title_score = 1.0 if has_core_target_title else (0.5 if has_secondary_title else 0.0)

    # Product vs Service Footprint Scanner
    SERVICE_KEYWORDS = {"it services", "consulting", "outsourcing", "systems integration", "tcs", "infosys", "wipro", "cognizant", "capgemini", "accenture"}
    has_product_company_exp = 0
    has_service_company_only = 0
    service_company_count = 0
    total_companies_worked = 0

    all_companies = [{"industry": profile.get("current_industry", ""), "name": profile.get("current_company", "")}]
    for job in career_history:
        all_companies.append({"industry": job.get("industry", ""), "name": job.get("company", "")})

    for comp in all_companies:
        ind = str(comp.get("industry", "")).lower()
        cname = str(comp.get("name", "")).lower()
        if not ind and not cname: continue
        total_companies_worked += 1
        
        if any(kw in ind for kw in SERVICE_KEYWORDS) or any(kw in cname for kw in SERVICE_KEYWORDS):
            service_company_count += 1
        else:
            has_product_company_exp = 1

    if total_companies_worked > 0 and service_company_count == total_companies_worked:
        has_service_company_only = 1

    # ----------------------------------------------------
    # LAYERS 6 - 10: METRIC TEXT SIGNALS & FOOTPRINTS
    # ----------------------------------------------------
    has_large_scale_system_exp = 1 if any(w in full_text for w in ["million users", "millions", "scale", "high throughput", "low latency", "streaming", "real-time", "realtime", "kafka", "spark", "distributed", "terabyte", "petabyte", "gb/day", "tb/day"]) else 0
    has_leadership_signal = 1 if any(w in full_text for w in ["led", "managed", "ownership", "headed", "architected", "mentored", "championed"]) else 0
    has_cross_functional_work_exp = 1 if any(w in full_text for w in ["stakeholders", "product manager", "recruiters", "cross functional", "collaborated", "worked closely with", "partnered with"]) else 0
    has_production_system_stability_exp = 1 if any(w in full_text for w in ["sla", "uptime", "on-call", "on call", "pagerduty", "incident", "monitoring", "observability"]) else 0

    # Layer 10 Open Source Signature Scanner
    has_open_source_signal = 0
    if any(w in full_text for w in ["github", "open source", "contributor", "maintainer", "pull request", "community"]):
        has_open_source_signal = 1
        if any(w in full_text for w in ["huggingface", "transformers", "langchain", "faiss", "llamaindex"]):
            has_open_source_signal = 2

    # ----------------------------------------------------
    # LAYERS 11 - 13: BEHAVIORAL ENGAGEMENT, NOTICE, AND LOCATION Fit
    # ----------------------------------------------------
    last_active_str = signals.get("last_active_date", "2026-01-01")
    try:
        last_active_date = datetime.strptime(last_active_str, "%Y-%m-%d")
        current_date_anchor = datetime(2026, 6, 13) 
        days_since_active = max(0, (current_date_anchor - last_active_date).days)
    except Exception:
        days_since_active = 30 

    # FIX: Generating missing behavioral and logistics variables
    # Normalizing behavioral metrics to a 0.0 - 1.0 scale
    behavioral_score = min(1.0, (response_rate * 0.4) + (interview_completion_rate * 0.4) + (open_to_work * 0.2))
    
    # Grading notice period (shorter is better)
    if notice_period <= 30:
        notice_score = 1.0
    elif notice_period <= 60:
        notice_score = 0.5
    else:
        notice_score = 0.0

    # Defaults for unspecified metrics
    location_fit_score = 1.0  # Defaulting to 1.0 as location isn't extracted from candidate_json
    honeypot_risk = 0  # Defaulting to 0 unless specific security logic is introduced

    # ----------------------------------------------------
    # CORE RANKING ALGORITHM WEIGHT MATRIX COMPOSITION
    # ----------------------------------------------------
    retrieval_strength = min(1.0, (retrieval_skill_count * 0.3) + (retrieval_keyword_hits * 0.15))
    ranking_strength = min(1.0, (ranking_skill_count * 0.5) + (any(w in full_text for w in RANKER_KW) * 0.3))
    vector_db_strength = min(1.0, vector_db_skill_count * 0.5)
    evaluation_strength = min(1.0, evaluation_keyword_hits * 0.4)
    github_normalized = min(1.0, github_score / 60.0)

    # Core Metric Tiers Aggregator
    total_score = (
        (15.0 * experience_fit_score) +
        (10.0 * title_score) +
        (20.0 * retrieval_strength) +
        (15.0 * ranking_strength) +
        (10.0 * vector_db_strength) +
        (10.0 * evaluation_strength) +
        (10.0 * has_production_ml_signal) +
        (10.0 * has_production_retrieval_signal) +
        (5.0  * has_large_scale_system_exp) +
        (5.0  * has_leadership_signal) +
        (5.0  * has_cross_functional_work_exp) +
        (5.0  * has_production_system_stability_exp) +
        (5.0  * (1.0 if has_open_source_signal >= 1 else 0.0)) +
        (5.0  * github_normalized) +
        (5.0  * behavioral_score) +
        (5.0  * notice_score) +
        (5.0  * location_fit_score)
    )

    # Apply strict negative conditional penalty layers
    if research_only_risk:
        total_score -= 15.0 
    if has_service_company_only:
        total_score *= 0.85 
    if honeypot_risk:
        total_score *= 0.40 

    normalized_final_score = (total_score / 135.0) * 100.0
    final_score = max(0.0, min(100.0, normalized_final_score))

    # ----------------------------------------------------
    # OUTPUT PAYLOAD INTERFACE
    # ----------------------------------------------------
    return {
        "candidate_id": candidate_json.get("candidate_id"),
        "total_match_score": round(final_score, 1),
        "hard_filters": {
            "experience_years": experience_years,
            "experience_fit_score": experience_fit_score,
            "has_production_ml_signal": has_production_ml_signal,
            "research_only_risk": research_only_risk,
            "honeypot_risk": honeypot_risk
        },
        "technical_match": {
            "retrieval_skill_count": retrieval_skill_count,
            "retrieval_keyword_hits": retrieval_keyword_hits,
            "vector_db_skill_count": vector_db_skill_count,
            "vector_db_keyword_hits": vector_db_keyword_hits,
            "ranking_skill_count": ranking_skill_count,
            "evaluation_keyword_hits": evaluation_keyword_hits
        },
        "production_signals": {
            "has_production_retrieval_signal": has_production_retrieval_signal,
            "has_large_scale_system_exp": has_large_scale_system_exp,
            "has_leadership_signal": has_leadership_signal,
            "has_cross_functional_work_exp": has_cross_functional_work_exp,
            "has_production_system_stability_exp": has_production_system_stability_exp,
            "has_open_source_signal": has_open_source_signal,
            "has_product_company_exp": has_product_company_exp,
            "has_service_company_only": has_service_company_only
        },
        "behavioral_and_logistics": {
            "github_score": github_score,
            "behavioral_score": round(behavioral_score, 2),
            "days_since_active": days_since_active,
            "notice_period_days": notice_period,
            "notice_score": notice_score,
            "location_fit_score": location_fit_score
        }
    }