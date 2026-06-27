"""
config.py — Master configuration for the Redrob Candidate Ranker
ALL constants, weights, ontologies, and JD definitions live here.
Change weights here, nowhere else.
"""

# ============================================================
# JOB DESCRIPTION — STRUCTURED DECOMPOSITION
# ============================================================

JD_FULL_TEXT = """
Senior ML AI Engineer Redrob recruiting platform Noida Pune India.
5 to 9 years experience production embeddings retrieval ranking systems.
Sentence transformers BGE E5 OpenAI embeddings dense retrieval.
Vector databases hybrid search Pinecone Weaviate Qdrant Milvus FAISS Elasticsearch OpenSearch Lucene.
Strong Python production quality code not notebooks.
Ranking evaluation NDCG MRR MAP A/B testing offline online evaluation.
LLM fine-tuning LoRA QLoRA PEFT retrieval augmented generation RAG.
Learning to rank XGBoost neural LTR.
HR tech recruiting marketplace recommendation systems.
Information retrieval semantic search candidate job matching.
Product company experience shipped to real users production deployment.
Active GitHub open source contributions AI ML NLP.
Mentoring growing team architecture design systems thinking.
Not pure research academic not consulting only TCS Infosys Wipro Accenture Cognizant.
Not computer vision speech robotics without NLP IR background.
Writes code actively not just architecture or tech lead.
"""

JD_CORE = """
Senior ML engineer production retrieval ranking recommendation system.
Embeddings FAISS Elasticsearch Pinecone vector search dense retrieval.
Python production deployment evaluation NDCG information retrieval.
Product company shipped real users 5 9 years experience NLP search.
"""

JD_SOFT = """
LLM fine-tuning LoRA QLoRA PEFT RAG. Learning to rank XGBoost.
HR tech recruiting marketplace distributed systems scale inference.
Open source contributions GitHub active coder.
"""

# Multiple JD chunks for multi-query embedding approach
JD_CHUNKS = {
    "core": JD_CORE,
    "full": JD_FULL_TEXT,
    "soft": JD_SOFT,
}

# ============================================================
# STAGE WEIGHTS — tune here
# ============================================================
STAGE_WEIGHTS = {
    "cross_encoder": 0.45,   # Most accurate — cross-encoder reranker
    "semantic_bge":  0.30,   # Bi-encoder semantic similarity
    "rule_features": 0.25,   # Engineered feature score
}

# ============================================================
# FEATURE WEIGHTS (must sum to 1.0)
# ============================================================
FEATURE_WEIGHTS = {
    # DOMAIN FIT — 32%
    "title_tier":               0.10,
    "career_trajectory_ml":     0.05,
    "product_company_ratio":    0.06,
    "retrieval_specificity":    0.07,
    "industry_relevance":       0.02,
    "career_progression_score": 0.02,  # NEW: seniority vs YoE consistency
    # SKILLS — 24%
    "weighted_skill_score":     0.09,
    "skill_depth":              0.04,
    "avg_assessment_score":     0.03,
    "skill_pair_synergy":       0.04,
    "skill_recency":            0.02,
    "github_activity":          0.03,
    "community_trust":          0.01,
    # EXPERIENCE — 22%
    "yoe_fit":                  0.06,
    "recent_ml_work":           0.04,
    "longest_ml_role_months":   0.04,
    "quantified_impact":        0.03,
    "career_stability":         0.01,
    "degree_field":             0.02,
    # BEHAVIORAL — 15%
    "recency_score":            0.04,
    "response_rate":            0.04,
    "open_to_work":             0.02,
    "notice_period_score":      0.02,
    "interview_completion":     0.01,
    "profile_trust":            0.01,
    "profile_completeness":     0.01,
    # LOGISTICS — 5%
    "location_fit":             0.02,
    "salary_fit":               0.02,
    # GROUP G: Conversion signals (uses previously ignored schema fields) — 3%
    "applications_submitted":   0.01,  # actively applying = high urgency
    "offer_acceptance_rate":    0.01,  # will they join after offer?
    "platform_seniority":       0.01,  # long-term platform engagement
}
assert abs(sum(FEATURE_WEIGHTS.values()) - 1.0) < 0.001, \
    f"Feature weights must sum to 1.0, got {sum(FEATURE_WEIGHTS.values()):.4f}"

# ============================================================
# TITLE TIER TAXONOMY
# ============================================================
TITLE_TIER_1 = [
    "machine learning engineer", "ml engineer", "senior ml", "staff ml",
    "principal ml", "ai engineer", "senior ai", "search engineer",
    "nlp engineer", "applied scientist", "research engineer",
    "recommendation engineer", "relevance engineer", "ranking engineer",
    "information retrieval", "applied ml", "senior machine learning",
    "lead ml", "lead ai", "ml platform", "ai platform",
]
TITLE_TIER_2 = [
    "data scientist", "senior data scientist", "senior engineer",
    "backend engineer", "software engineer", "platform engineer",
    "data engineer", "full stack engineer", "senior software",
    "senior backend", "senior data", "ml ops", "mlops",
    "systems engineer", "senior systems",
]
TITLE_TIER_3 = [
    "junior ml", "junior data", "associate data", "associate engineer",
    "data analyst", "analytics engineer", "bi engineer",
]
TITLE_IRRELEVANT = [
    "hr manager", "hr executive", "human resources", "recruiter",
    "content writer", "copywriter", "graphic designer", "ux designer",
    "ui designer", "product designer", "accountant", "finance",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "sales executive", "sales manager", "business development",
    "marketing manager", "marketing executive", "seo", "social media",
    "operations manager", "operations executive", "teacher", "lecturer",
    "nurse", "doctor", "customer support", "customer service",
    "project manager", "scrum master", "business analyst",
]

# ============================================================
# SKILL ONTOLOGY — WITH TIERS AND SYNONYMS
# ============================================================
SKILL_ONTOLOGY = {
    "vector_search": {
        "tier": "CRITICAL",
        "weight": 1.0,
        "matches": {
            "faiss", "annoy", "hnswlib", "nmslib", "scann", "vector search",
            "dense retrieval", "approximate nearest neighbor", "ann search",
            "semantic search", "pinecone", "weaviate", "qdrant", "milvus",
            "chroma", "pgvector", "vespa", "marqo", "typesense",
        }
    },
    "text_search": {
        "tier": "CRITICAL",
        "weight": 0.9,
        "matches": {
            "elasticsearch", "opensearch", "solr", "lucene", "bm25", "tf-idf",
            "inverted index", "full text search", "hybrid search",
            "sparse retrieval", "keyword search", "whoosh",
        }
    },
    "embeddings": {
        "tier": "CRITICAL",
        "weight": 1.0,
        "matches": {
            "sentence-transformers", "sbert", "bge", "e5", "ada-002",
            "text-embedding", "mpnet", "bert", "roberta", "distilbert",
            "minilm", "bi-encoder", "dense encoder", "contrastive learning",
            "siamese network", "embedding model", "word2vec", "fasttext",
            "glove", "doc2vec",
        }
    },
    "ranking_recsys": {
        "tier": "CRITICAL",
        "weight": 1.0,
        "matches": {
            "ranking", "recommendation", "collaborative filtering",
            "learning to rank", "ltr", "reranking", "cross-encoder",
            "listwise", "pairwise", "pointwise", "lambdamart", "ranknet",
            "ndcg", "mrr", "relevance", "retrieval", "matrix factorization",
            "item2vec", "two-tower", "candidate generation",
        }
    },
    "nlp_frameworks": {
        "tier": "IMPORTANT",
        "weight": 0.75,
        "matches": {
            "nlp", "natural language processing", "text classification",
            "ner", "named entity recognition", "spacy", "huggingface",
            "transformers", "language model", "llm", "gpt", "fine-tuning",
            "lora", "qlora", "peft", "rag", "retrieval augmented",
            "tokenization", "text mining", "nltk",
        }
    },
    "ml_frameworks": {
        "tier": "IMPORTANT",
        "weight": 0.7,
        "matches": {
            "pytorch", "tensorflow", "keras", "jax", "sklearn",
            "scikit-learn", "xgboost", "lightgbm", "catboost",
            "numpy", "pandas",
        }
    },
    "evaluation": {
        "tier": "IMPORTANT",
        "weight": 0.8,
        "matches": {
            "a/b testing", "a/b test", "ab testing", "ndcg", "mrr", "map",
            "offline evaluation", "online evaluation", "relevance judgment",
            "human annotation", "click-through rate", "conversion rate",
        }
    },
    "mlops_cloud": {
        "tier": "NICE_TO_HAVE",
        "weight": 0.4,
        "matches": {
            "aws", "gcp", "azure", "docker", "kubernetes", "k8s", "mlflow",
            "ray", "airflow", "kubeflow", "sagemaker", "vertex ai",
            "mlops", "model serving", "triton", "torchserve",
        }
    },
}

# ============================================================
# CONSULTING GIANTS — Career consulting-only penalty
# ============================================================
CONSULTING_GIANTS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "tech mahindra", "hcl technologies",
    "hcl", "mphasis", "hexaware", "l&t infotech", "ltimindtree",
    "persistent systems", "mastech", "zensar", "cyient",
    "niit technologies", "mindtree", "firstsource", "wns global",
    "wns", "genpact", "syntel", "igate", "patni", "sasken",
    "kpit", "atos", "dxc technology", "unisys", "birlasoft",
    "mphasis", "sonata software", "mphasis", "oracle financial",
}

# ============================================================
# RETRIEVAL-SPECIFIC TERMS (NOT generic ML)
# ============================================================
RETRIEVAL_SPECIFIC_TERMS = {
    "retrieval", "ranking", "recommendation engine", "search system",
    "document ranking", "relevance scoring", "content retrieval",
    "personalization engine", "similarity matching", "candidate matching",
    "information filtering", "product ranking", "feed algorithm",
    "content recommendation", "job matching", "text similarity",
    "semantic matching", "ranking algorithm", "search relevance",
    "vector representation", "dense representation", "inverted index",
    "shipped to production", "scaled to millions", "serving millions",
    "real users", "production system", "live system", "recommendation system",
    "retrieval system", "search infrastructure", "relevance engine",
    "reranking", "cross-encoder", "bi-encoder", "dual encoder",
    "approximate nearest neighbor", "nearest neighbor search",
}

# ============================================================
# LOCATION SCORING
# ============================================================
PREFERRED_CITIES = {
    "noida", "gurgaon", "gurugram", "delhi", "ncr", "new delhi",
    "pune", "pimpri", "pcmc", "greater noida",
}
ACCEPTABLE_CITIES = {
    "bangalore", "bengaluru", "mumbai", "hyderabad", "chennai",
    "kolkata", "ahmedabad", "jaipur", "chandigarh", "indore",
    "nagpur", "surat", "bhopal", "kochi", "thiruvananthapuram",
    "coimbatore", "vadodara", "nashik", "navi mumbai", "thane",
}

# ============================================================
# WRONG DOMAIN (CV/Speech/Robotics without NLP/IR)
# ============================================================
WRONG_DOMAIN_SIGNALS = {
    "computer vision", "object detection", "image segmentation",
    "image classification", "yolo", "cnn", "convolutional",
    "speech recognition", "asr", "tts", "text to speech",
    "robotics", "autonomous driving", "slam", "ros", "control system",
}

WRONG_DOMAIN_MITIGATING = {
    "nlp", "retrieval", "ranking", "recommendation", "information retrieval",
    "search", "embedding", "transformer", "bert", "natural language",
}

# ============================================================
# HONEYPOT DETECTION THRESHOLDS
# ============================================================
HONEYPOT_CONFIG = {
    "exp_mismatch_months_threshold": 42,  # > 3.5 year gap is suspicious
    "skill_duration_buffer_months": 6,
    "edu_career_overlap_years": 3,
    "future_cert_tolerance_years": 0,
    "min_flags_for_honeypot": 2,
    "strong_flag_instant_hp": True,
}

STRONG_HONEYPOT_FLAGS = {
    "salary_range_inverted", "future_job", "career_after_signup"
}
MEDIUM_HONEYPOT_FLAGS = {
    "exp_mismatch", "skill_duration_impossible", "future_cert"
}
WEAK_HONEYPOT_FLAGS = {
    "instant_expert", "edu_career_overlap"
}

# ============================================================
# SCORING BOUNDS
# ============================================================
BEHAVIORAL_MULT_MIN = 0.35
BEHAVIORAL_MULT_MAX = 1.25
SCORE_MIN = 0.0
SCORE_MAX = 1.0

# ============================================================
# PIPELINE STAGE SIZES
# ============================================================
STAGE1_OUTPUT = 25000    # After hard filter
STAGE2_TFIDF_TOP = 6000  # After BM25 pre-filter
STAGE2_SEMANTIC_TOP = 2000  # After bi-encoder
STAGE3_CROSSENCODER_TOP = 500  # Cross-encoder reranks top-500
FINAL_OUTPUT = 100

# ============================================================
# EMBEDDING MODEL SELECTION
# ============================================================
# GPU (Colab): Use large model for maximum quality
EMBED_MODEL_GPU = "BAAI/bge-large-en-v1.5"       # 335M params, best quality
EMBED_MODEL_CPU = "BAAI/bge-small-en-v1.5"        # 33M params, fast CPU

# Cross-encoder for reranking (GPU only — too slow for CPU on 2000 candidates)
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-large"   # Best available reranker

# BGE instruction prefix (improves quality significantly for BGE models)
BGE_QUERY_INSTRUCTION = "Represent this query for searching relevant documents: "
BGE_DOC_INSTRUCTION = "Represent this document for retrieval: "
