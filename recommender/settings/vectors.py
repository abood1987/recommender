# ----------------------------------
# EXTRACTORS
EXTRACTOR_LLM = {
    "class": "recommender_core.extractor.LLMExtractor",
    "configuration": {
        "extract_prompt": "Extract professional skills from the following text: %s. Return only skills, comma-separated.",
        "description_prompt": "Generate a detailed professional description for the skill: %s.",
        "occupation_extract_prompt": "Extract and describe a single professional occupation from the following text: %s. Return only a short, formal description of the occupation.",
    },
}
EXTRACTOR_HYBRID = {
    "class": "recommender_core.extractor.HybridExtractor",
    "configuration": {
        "extract_prompt": "Extract professional skills from the following text: %s. Return only skills, comma-separated.",
        "description_prompt": "Generate a detailed professional description for the skill: %s.",
        "occupation_extract_prompt": "Extract and describe a single professional occupation from the following text: %s. Return only a short, formal description of the occupation.",
    },
}
EXTRACTOR_NER_JOBBERT = {
    "class": "recommender_core.extractor.NERExtractor",
    "configuration": {
        "skills_model": "jjzha/jobbert_skill_extraction",
        "knowledge_model": "jjzha/jobbert_knowledge_extraction",
    },
}
EXTRACTOR_NER_ESCOXLMR = {
    "class": "recommender_core.extractor.NERExtractor",
    "configuration": {
        "skills_model": "jjzha/escoxlmr_skill_extraction",
        "knowledge_model": "jjzha/escoxlmr_knowledge_extraction",
    },
}
EXTRACTOR_EMBEDDING_SIMILARITY = {
    "class": "recommender_core.extractor.EmbeddingSimilarityExtractor ",
    "configuration": {},
}
EXTRACTOR_SPLIT = {
    "class": "recommender_core.extractor.SplitExtraction",
    "configuration": {},
}
# ---------------------------------------------
# ---------------------------------------------
# LLM
LLM_FLAN_T5 = {
    "class": "recommender_core.llm.FlanT5Model",
    "configuration": {
        "model_name": "google/flan-t5-large",
        "model_path": "C:\\Users\\abd19\\PycharmProjects\\recommender\\data\\models\\google-flan-t5-large",
    },
}
LLM_FLAN_T5_FT = {
    "class": "recommender_core.llm.FlanT5Model",
    "configuration": {
        "model_name": "abd1987/esco-flan-t5-large",
        "model_path": "C:\\Users\\abd19\\PycharmProjects\\recommender\\data\\models\\flan-t5-esco",
    },
}
# ---------------------------------------------
# ---------------------------------------------
# Matchers

MATCHER_BINARY_VECTOR = {
    # Save the vectors in DB to speed up the process
    "class": "recommender_core.matcher.BinaryVectorMatcher",
    "configuration": {
        "threshold": 0.4,
        "include_broader": True,
        "top_k": None,
    },
}
MATCHER_EMBEDDINGS = {
    # Save the embeddings in DB to speed up the process
    "class": "recommender_core.matcher.EmbeddingMatcher",
    "configuration": {
        "threshold": 0.7,
        "include_broader": True,
        "top_k": None,
    },
}
MATCHER_FUZZY = {
    "class": "recommender_core.matcher.FuzzyMatcher",
    "configuration": {
        "threshold": 0.6,
        "include_broader": True,
        "fuzzy_threshold": 60,  # [0 - 100]
        "top_k": None,
    },
}
MATCHER_OVERLAP = {
    "class": "recommender_core.matcher.OverlapMatcher",
    "configuration": {
        "threshold": 0.4,
        "include_broader": True,
        "top_k": None,
    },
}
MATCHER_TF_IDF = {
    "class": "recommender_core.matcher.TFIDFMatcher",
    "configuration": {
        "threshold": 0.5,
        "include_broader": True,
        "top_k": None,
    },
}


VECTOR_SETTINGS = {
    # Configuration is passed directly to the model during initialization.
    "max_distance": 0.4,
    "top_k": 5,
    "embeddings": {
        "class": "recommender_core.embeddings.SentenceTransformerModel",
        "configuration": {
            # "model_name": "paraphrase-mpnet-base-v2",
            "model_name": "abd1987/esco-context-skill-extraction",
            "model_path": "C:\\Users\\abd19\\PycharmProjects\\recommender\\data\\models\\esco-embeddings"
        },
    },
    "llm": LLM_FLAN_T5_FT,
    "extractor": EXTRACTOR_HYBRID,
    "matcher": MATCHER_OVERLAP
}
