VECTOR_SETTINGS = {
    "embeddings": {
        "class": "recommender_core.embeddings.SentenceTransformerModel",
        # Configuration is passed directly to the embeddings model class during initialization.
        "configuration": {
            "model_name": "sentence-transformers/all-mpnet-base-v2",
        },
    },
    "matcher": {
        "class": "recommender_core.embeddings.T5MatcherModel",
        # Configuration is passed directly to the matcher model class during initialization.
        "configuration": {
            "model_name": "google/flan-t5-large",
            "skill_description_prompt": "Generate a detailed professional description for the skill: %s.",
            "extract_skills_prompt": "Extract individual skills from the following text. Ensure each skill is clearly identified, professional, and in a normalized format:\n%s\nList of skills:"
        },
    },
}
