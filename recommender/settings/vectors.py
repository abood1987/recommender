EXTRACTOR_LLM = {
    "class": "recommender_core.extractor.LLMExtractor",
    "configuration": {
        "extract_prompt": "Extract professional skills from the following text: %s. Return only skills, comma-separated.",
        "description_prompt": "Generate a detailed professional description for the skill: %s.",
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

EXTRACTOR_SIMPLE = {
    "class": "recommender_core.extractor.SimpleExtraction",
    "configuration": {},
}

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
        "model_name": "google/flan-t5-large",
        "model_path": "C:\\Users\\abd19\\PycharmProjects\\recommender\\data\\models\\google-flan-t5-esco",
    },
}


VECTOR_SETTINGS = {
    # Configuration is passed directly to the model during initialization.
    "max_distance": 0.45,
    "embeddings": {
        "class": "recommender_core.embeddings.SentenceTransformerModel",
        "configuration": {
            # "model_name": "paraphrase-mpnet-base-v2",
            "model_name": "abd1987/esco-context-skill-extraction",
            "model_path": "C:\\Users\\abd19\\PycharmProjects\\recommender\\data\\models\\paraphrase-mpnet-base-v2"
        },
    },
    "llm": {
        "class": "recommender_core.llm.FlanT5Model",
        "configuration": {
            "model_name": "google/flan-t5-large",
            "model_path": "C:\\Users\\abd19\\PycharmProjects\\recommender\\data\\models\\google-flan-t5-esco",
        },
    },
    "extractor": EXTRACTOR_SIMPLE,
}
